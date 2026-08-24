#include "atpg.h"

#include <stdexcept>

void ATPG::set_decision_policy(
    const shared_ptr<smartatpg::DecisionPolicy> &policy) {
  decision_policy = policy;
}

void ATPG::enable_rl_inference(const string &embedding_path,
                               const string &actor_path) {
  decision_policy = make_shared<smartatpg::NativeActorPolicy>(
      embedding_path, actor_path, smartatpg::fnv1a_file_hash(filename));
}

void ATPG::disable_rl_policy() {
  decision_policy.reset();
  rl_podem_episode_active = false;
}

int ATPG::choose_policy_candidate(smartatpg::DecisionMode mode,
                                  const string &objective_name,
                                  const int &objective_value,
                                  const vector<string> &candidate_names) {
  if (!decision_policy || candidate_names.size() <= 1) {
    return -1;
  }

  smartatpg::DecisionRequest request;
  request.mode = mode;
  request.objective_name = objective_name;
  request.objective_value = objective_value;
  request.candidate_names = candidate_names;
  request.sequence = ++rl_decision_sequence;
  request.fault_id = current_rl_fault_id;
  request.backtracks = no_of_backtracks;

  const int selected = decision_policy->select(request);
  if (selected < 0 || static_cast<size_t>(selected) >= candidate_names.size()) {
    throw runtime_error("Policy returned invalid " +
                        smartatpg::decision_mode_name(mode) +
                        " candidate index " + to_string(selected) +
                        " for " + to_string(candidate_names.size()) +
                        " candidates");
  }
  if (mode == smartatpg::DecisionMode::BACKTRACE) {
    last_backtrace_decision_sequence = request.sequence;
  }
  return selected;
}

string ATPG::fault_identifier(const fptr fault) const {
  string id = fault->node ? fault->node->name : "<primary-input>";
  id += (fault->io == GO ? ":GO:" : ":GI" + to_string(fault->index) + ":");
  id += fault->fault_type == STUCK1 ? "sa1" : "sa0";
  return id;
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
  result.decisions = rl_decision_sequence;
  rl_podem_episode_active = false;
  decision_policy->on_episode_end(result);
}
