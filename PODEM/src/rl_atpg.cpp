#include "atpg.h"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

void ATPG::set_decision_policy(
    const shared_ptr<smartatpg::DecisionPolicy> &policy) {
  decision_policy = policy;
}

void ATPG::enable_rl_inference(const string &embedding_path,
                               const string &actor_path) {
  shared_ptr<smartatpg::DecisionPolicy> policy =
      make_shared<smartatpg::NativeActorPolicy>(
      embedding_path, actor_path, smartatpg::fnv1a_file_hash(filename),
      rl_gate_names_by_id);
  if ((smartatpg::rl_mode_enables(rl_mode,
                                  smartatpg::DecisionMode::BACKTRACE) &&
       !policy->supports(smartatpg::DecisionMode::BACKTRACE)) ||
      (smartatpg::rl_mode_enables(rl_mode,
                                  smartatpg::DecisionMode::PROPAGATION) &&
       !policy->supports(smartatpg::DecisionMode::PROPAGATION))) {
    throw runtime_error("Selected RL mode is not supported by this actor");
  }
  decision_policy = policy;
}

void ATPG::set_rl_mode(const string &value) {
  const smartatpg::RlMode selected = smartatpg::parse_rl_mode(value);
  if (decision_policy &&
      ((smartatpg::rl_mode_enables(selected,
                                   smartatpg::DecisionMode::BACKTRACE) &&
        !decision_policy->supports(smartatpg::DecisionMode::BACKTRACE)) ||
       (smartatpg::rl_mode_enables(selected,
                                   smartatpg::DecisionMode::PROPAGATION) &&
        !decision_policy->supports(smartatpg::DecisionMode::PROPAGATION)))) {
    throw runtime_error("Selected RL mode is not supported by this actor");
  }
  rl_mode = selected;
}

void ATPG::disable_rl_policy() {
  decision_policy.reset();
  rl_podem_episode_active = false;
}

void ATPG::retain_faults(const vector<string> &fault_ids) {
  if (fault_ids.empty()) {
    throw runtime_error("Requested fault list must not be empty");
  }
  unordered_map<string, fptr> faults_by_id;
  for (fptr fault : flist_undetect) {
    const string id = fault_identifier(fault);
    if (!faults_by_id.emplace(id, fault).second) {
      throw runtime_error("Duplicate generated fault identifier: " + id);
    }
  }

  unordered_set<string> requested;
  vector<fptr> selected;
  selected.reserve(fault_ids.size());
  for (const string &id : fault_ids) {
    if (!requested.insert(id).second) {
      throw runtime_error("Duplicate requested fault identifier: " + id);
    }
    const auto found = faults_by_id.find(id);
    if (found == faults_by_id.end()) {
      throw runtime_error("Unknown requested fault identifier: " + id);
    }
    selected.push_back(found->second);
  }

  flist_undetect.clear();
  for (auto pos = selected.rbegin(); pos != selected.rend(); ++pos) {
    (*pos)->test_tried = false;
    (*pos)->detect = false;
    flist_undetect.push_front(*pos);
  }
}

int ATPG::choose_policy_candidate(smartatpg::DecisionMode mode,
                                  const wptr objective_wire,
                                  const int &objective_value,
                                  const vector<wptr> &candidate_wires) {
  if (!decision_policy || candidate_wires.size() <= 1 ||
      !rl_enabled_for(mode)) {
    return -1;
  }

  smartatpg::DecisionRequest request;
  request.mode = mode;
  if (objective_wire) {
    request.objective_id = objective_wire->rl_gate_id;
  }
  request.objective_value = objective_value;
  rl_candidate_ids_scratch.clear();
  rl_candidate_ids_scratch.reserve(candidate_wires.size());
  for (wptr candidate : candidate_wires) {
    rl_candidate_ids_scratch.push_back(candidate->rl_gate_id);
  }
  request.candidate_ids = rl_candidate_ids_scratch.data();
  request.candidate_count = rl_candidate_ids_scratch.size();
  if (mode == smartatpg::DecisionMode::BACKTRACE && objective_wire &&
      !objective_wire->inode.empty()) {
    const int gate_type = objective_wire->inode.front()->type;
    const bool easiest =
        ((gate_type == OR || gate_type == NAND) && objective_value) ||
        ((gate_type == NOR || gate_type == AND) && !objective_value);
    request.heuristic_action =
        easiest ? 0 : static_cast<int>(candidate_wires.size() - 1);
  }
  if (decision_policy->needs_gate_names()) {
    if (objective_wire) {
      request.objective_name = objective_wire->name;
    }
    request.candidate_names.reserve(candidate_wires.size());
    for (wptr candidate : candidate_wires) {
      request.candidate_names.push_back(candidate->name);
    }
    request.fault_id = current_rl_fault_id;
  }
  request.sequence = ++rl_decision_sequence;
  request.backtracks = no_of_backtracks;

  const int selected = decision_policy->select(request);
  if (selected < 0 ||
      static_cast<size_t>(selected) >= candidate_wires.size()) {
    throw runtime_error("Policy returned invalid " +
                        smartatpg::decision_mode_name(mode) +
                        " candidate index " + to_string(selected) +
                        " for " + to_string(candidate_wires.size()) +
                        " candidates");
  }
  last_policy_decision_sequence = request.sequence;
  return selected;
}

bool ATPG::rl_enabled_for(smartatpg::DecisionMode mode) const {
  return decision_policy && smartatpg::rl_mode_enables(rl_mode, mode);
}

string ATPG::fault_identifier(const fptr fault) const {
  if (!fault->external_id.empty()) {
    return fault->external_id;
  }
  string id = fault->node ? fault->node->name : "<primary-input>";
  id += (fault->io == GO ? ":GO:" : ":GI" + to_string(fault->index) + ":");
  id += fault->fault_type == STUCK1 ? "sa1" : "sa0";
  return id;
}

void ATPG::notify_pi_result(bool detected) {
  rl_pi_visits += rl_pending_pi_assignments;
  rl_pending_pi_assignments = 0;
  const bool aborted_at_limit = !detected && no_of_backtracks >= backtrack_limit;
  if (decision_policy && decision_policy->wants_training_events() && !detected &&
      !aborted_at_limit) {
    decision_policy->on_pi_not_done(last_policy_decision_sequence,
                                    no_of_backtracks, rl_pi_visits);
  }
}

void ATPG::notify_episode_end(const int &outcome) {
  if (!decision_policy) {
    rl_podem_episode_active = false;
    return;
  }
  smartatpg::EpisodeResult result;
  result.fault_id = current_rl_fault_id;
  result.outcome = outcome;
  result.backtracks = no_of_backtracks;
  result.backtrace_steps = episode_backtrace_steps;
  result.pi_visits = rl_pi_visits;
  result.decisions = rl_decision_sequence;
  rl_podem_episode_active = false;
  decision_policy->on_episode_end(result);
}
