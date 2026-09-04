#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "atpg.h"
#include "rl_policy.h"

#include <memory>
#include <string>

namespace py = pybind11;

namespace {

class PythonDecisionPolicy : public smartatpg::DecisionPolicy {
public:
  PythonDecisionPolicy(py::function decision_callback, py::object event_callback)
      : decision_callback_(decision_callback), event_callback_(event_callback),
        wants_training_events_(!event_callback.is_none()) {}

  bool wants_training_events() const override {
    return wants_training_events_;
  }

  int select(const smartatpg::DecisionRequest &request) override {
    py::gil_scoped_acquire acquire;
    py::dict data;
    data["mode"] = smartatpg::decision_mode_name(request.mode);
    data["objective_name"] = request.objective_name;
    data["objective_value"] = request.objective_value;
    data["candidate_names"] = request.candidate_names;
    data["action_mask"] = std::vector<bool>{request.action_mask[0],
                                             request.action_mask[1]};
    data["heuristic_action"] = request.heuristic_action;
    data["sequence"] = request.sequence;
    data["fault_id"] = request.fault_id;
    data["backtracks"] = request.backtracks;
    ++decisions_;
    return decision_callback_(data).cast<int>();
  }

  void on_episode_start(const std::string &fault_id) override {
    emit_event("episode_start", fault_id, 0, 0, 0, 0, 0);
  }

  void on_backtrack(unsigned long decision_sequence) override {
    emit_event("backtrack", "", 0, 0, decision_sequence, 0, 0);
  }

  void on_backtrace_step(unsigned long decision_sequence) override {
    emit_event("backtrace_step", "", 0, 0, decision_sequence, 0, 0);
  }

  void on_pi_not_done(unsigned long decision_sequence, int backtracks,
                      unsigned long pi_visits) override {
    emit_event("pi_not_done", "", 0, backtracks, decision_sequence, 0,
               pi_visits);
  }

  void on_episode_end(const smartatpg::EpisodeResult &result) override {
    ++episodes_;
    if (result.outcome == TRUE) {
      ++detected_;
    } else if (result.outcome == FALSE) {
      ++redundant_;
    } else {
      ++aborted_;
    }
    backtracks_ += result.backtracks;
    backtrace_steps_ += result.backtrace_steps;
    pi_visits_ += result.pi_visits;
    emit_event("episode_end", result.fault_id, result.outcome,
               result.backtracks, result.decisions, result.backtrace_steps,
               result.pi_visits);
  }

  py::dict summary() const {
    py::dict result;
    result["episodes"] = episodes_;
    result["detected"] = detected_;
    result["redundant"] = redundant_;
    result["aborted"] = aborted_;
    result["decisions"] = decisions_;
    result["backtracks"] = backtracks_;
    result["backtrace_steps"] = backtrace_steps_;
    result["pi_visits"] = pi_visits_;
    return result;
  }

private:
  void emit_event(const char *event, const std::string &fault_id, int outcome,
                  int backtracks, unsigned long sequence,
                  unsigned long backtrace_steps, unsigned long pi_visits) {
    if (!wants_training_events_) {
      return;
    }
    py::gil_scoped_acquire acquire;
    py::dict data;
    data["event"] = event;
    if (!fault_id.empty()) {
      data["fault_id"] = fault_id;
    }
    if (std::string(event) == "episode_end") {
      data["outcome"] = outcome;
      data["backtracks"] = backtracks;
      data["decisions"] = sequence;
      data["backtrace_steps"] = backtrace_steps;
      data["pi_visits"] = pi_visits;
    } else if (std::string(event) == "backtrack") {
      data["decision_sequence"] = sequence;
    } else if (std::string(event) == "backtrace_step") {
      data["decision_sequence"] = sequence;
    } else if (std::string(event) == "pi_not_done") {
      data["decision_sequence"] = sequence;
      data["backtracks"] = backtracks;
      data["pi_visits"] = pi_visits;
    }
    event_callback_(data);
  }

  py::function decision_callback_;
  py::object event_callback_;
  bool wants_training_events_ = false;
  unsigned long episodes_ = 0;
  unsigned long detected_ = 0;
  unsigned long redundant_ = 0;
  unsigned long aborted_ = 0;
  unsigned long decisions_ = 0;
  unsigned long backtracks_ = 0;
  unsigned long backtrace_steps_ = 0;
  unsigned long pi_visits_ = 0;
};

py::dict run_stuck_at(const std::string &circuit_path,
                      py::function decision_callback,
                      py::object event_callback, int backtrack_limit,
                      int seed, py::object fault_ids, bool quiet,
                      const std::string &rl_mode,
                      const std::string &fault_map_path) {
  const bool has_fault_filter = !fault_ids.is_none();
  const std::vector<std::string> selected_faults = has_fault_filter
      ? fault_ids.cast<std::vector<std::string> >()
      : std::vector<std::string>();
  std::shared_ptr<PythonDecisionPolicy> policy =
      std::make_shared<PythonDecisionPolicy>(decision_callback, event_callback);
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_backtrack_limit(backtrack_limit);
  atpg.set_seed(seed);
  atpg.set_total_attempt_num(1);
  atpg.set_SAF_atpg(true);
  atpg.set_rl_mode(rl_mode);
  atpg.set_decision_policy(policy);
  atpg.set_quiet(quiet);
  atpg.set_fault_map_path(fault_map_path);

  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    if (has_fault_filter) {
      atpg.retain_faults(selected_faults);
      atpg.set_drop_detected_faults(false);
    }
    atpg.test();
    if (!has_fault_filter) {
      atpg.compute_fault_coverage();
    }
  }
  atpg.disable_rl_policy();
  return policy->summary();
}

py::list profile_stuck_at(const std::string &circuit_path,
                          int backtrack_limit, int seed,
                          const std::string &fault_map_path) {
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_backtrack_limit(backtrack_limit);
  atpg.set_seed(seed);
  atpg.set_total_attempt_num(1);
  atpg.set_SAF_atpg(true);
  atpg.set_drop_detected_faults(false);
  atpg.set_collect_fault_profiles(true);
  atpg.set_quiet(true);
  atpg.set_fault_map_path(fault_map_path);

  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    atpg.test();
  }

  py::list result;
  for (const ATPG::FaultProfile &profile : atpg.get_fault_profiles()) {
    py::dict item;
    item["fault_id"] = profile.fault_id;
    item["outcome"] = profile.outcome;
    item["backtracks"] = profile.backtracks;
    item["backtrace_steps"] = profile.backtrace_steps;
    result.append(item);
  }
  return result;
}

py::dict catalog_stuck_at(const std::string &circuit_path,
                          const std::string &fault_map_path) {
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_SAF_atpg(true);
  atpg.set_quiet(true);
  atpg.set_fault_map_path(fault_map_path);

  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
  }

  py::list faults;
  for (const ATPG::FaultCatalogEntry &entry : atpg.get_fault_catalog()) {
    py::dict item;
    item["fault_id"] = entry.fault_id;
    item["node_name"] = entry.node_name;
    item["input_wire_name"] = entry.input_wire_name;
    item["io"] = entry.io;
    item["input_index"] = entry.input_index;
    item["input_occurrence"] = entry.input_occurrence;
    item["fault_type"] = entry.fault_type;
    item["eqv_fault_num"] = entry.eqv_fault_num;
    faults.append(item);
  }
  py::dict result;
  result["faults"] = faults;
  result["uncollapsed_total"] = atpg.get_uncollapsed_fault_count();
  return result;
}

std::vector<float> score_actor_v2(const std::string &actor_path,
                                  const std::vector<float> &objective,
                                  int objective_value) {
  smartatpg::ActorModel actor;
  actor.load(actor_path);
  return actor.backtrace_action_logits(objective, objective_value);
}

void validate_actor_artifacts(const std::string &embedding_path,
                             const std::string &actor_path,
                             const std::string &circuit_hash,
                             const std::vector<std::string> &names,
                             const std::string &backend) {
  smartatpg::NativeActorPolicy policy(embedding_path, actor_path, circuit_hash, names, backend);
}

} // namespace

PYBIND11_MODULE(cpp_podem, module) {
  module.doc() = "Python training bridge for the C++ PODEM engine";
  module.def("run_stuck_at", &run_stuck_at, py::arg("circuit_path"),
             py::arg("decision_callback"),
             py::arg("event_callback") = py::none(),
             py::arg("backtrack_limit") = 97, py::arg("seed") = 14,
             py::arg("fault_ids") = py::none(), py::arg("quiet") = false,
             py::arg("rl_mode") = "backtrace_rl",
             py::arg("fault_map_path") = "");
  module.def("profile_stuck_at", &profile_stuck_at,
             py::arg("circuit_path"), py::arg("backtrack_limit") = 97,
             py::arg("seed") = 14, py::arg("fault_map_path") = "");
  module.def("catalog_stuck_at", &catalog_stuck_at,
             py::arg("circuit_path"), py::arg("fault_map_path") = "");
  module.def("score_actor_v2", &score_actor_v2, py::arg("actor_path"),
             py::arg("objective"), py::arg("objective_value"));
  module.def("validate_actor_artifacts", &validate_actor_artifacts,
             py::arg("embedding_path"), py::arg("actor_path"), py::arg("circuit_hash"),
             py::arg("names"), py::arg("backend") = "");
}
