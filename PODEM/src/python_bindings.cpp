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
      : decision_callback_(decision_callback), event_callback_(event_callback) {}

  int select(const smartatpg::DecisionRequest &request) override {
    py::gil_scoped_acquire acquire;
    py::dict data;
    data["mode"] = smartatpg::decision_mode_name(request.mode);
    data["objective_name"] = request.objective_name;
    data["objective_value"] = request.objective_value;
    data["candidate_names"] = request.candidate_names;
    data["sequence"] = request.sequence;
    data["fault_id"] = request.fault_id;
    data["backtracks"] = request.backtracks;
    ++decisions_;
    return decision_callback_(data).cast<int>();
  }

  void on_episode_start(const std::string &fault_id) override {
    emit_event("episode_start", fault_id, 0, 0, 0);
  }

  void on_backtrack(unsigned long decision_sequence) override {
    ++backtracks_;
    emit_event("backtrack", "", 0, 0, decision_sequence);
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
    emit_event("episode_end", result.fault_id, result.outcome,
               result.backtracks, result.decisions);
  }

  py::dict summary() const {
    py::dict result;
    result["episodes"] = episodes_;
    result["detected"] = detected_;
    result["redundant"] = redundant_;
    result["aborted"] = aborted_;
    result["decisions"] = decisions_;
    result["backtracks"] = backtracks_;
    return result;
  }

private:
  void emit_event(const char *event, const std::string &fault_id, int outcome,
                  int backtracks, unsigned long sequence) {
    py::gil_scoped_acquire acquire;
    if (event_callback_.is_none()) {
      return;
    }
    py::dict data;
    data["event"] = event;
    if (!fault_id.empty()) {
      data["fault_id"] = fault_id;
    }
    if (std::string(event) == "episode_end") {
      data["outcome"] = outcome;
      data["backtracks"] = backtracks;
      data["decisions"] = sequence;
    } else if (std::string(event) == "backtrack") {
      data["decision_sequence"] = sequence;
    }
    event_callback_(data);
  }

  py::function decision_callback_;
  py::object event_callback_;
  unsigned long episodes_ = 0;
  unsigned long detected_ = 0;
  unsigned long redundant_ = 0;
  unsigned long aborted_ = 0;
  unsigned long decisions_ = 0;
  unsigned long backtracks_ = 0;
};

py::dict run_stuck_at(const std::string &circuit_path,
                      py::function decision_callback,
                      py::object event_callback, int backtrack_limit,
                      int seed) {
  std::shared_ptr<PythonDecisionPolicy> policy =
      std::make_shared<PythonDecisionPolicy>(decision_callback, event_callback);
  ATPG atpg;
  atpg.set_backtrack_limit(backtrack_limit);
  atpg.set_seed(seed);
  atpg.set_total_attempt_num(1);
  atpg.set_SAF_atpg(true);
  atpg.set_decision_policy(policy);

  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    atpg.test();
    atpg.compute_fault_coverage();
  }
  atpg.disable_rl_policy();
  return policy->summary();
}

std::vector<float> score_actor(
    const std::string &actor_path, const std::string &mode,
    const std::vector<float> &objective,
    const std::vector<std::vector<float> > &candidates) {
  smartatpg::ActorModel actor;
  actor.load(actor_path);
  if (mode == "backtrace") {
    return actor.backtrace_logits(objective, candidates);
  }
  if (mode == "propagation") {
    return actor.propagation_logits(candidates);
  }
  throw std::runtime_error("Unknown actor mode: " + mode);
}

} // namespace

PYBIND11_MODULE(cpp_podem, module) {
  module.doc() = "Python training bridge for the C++ PODEM engine";
  module.def("run_stuck_at", &run_stuck_at, py::arg("circuit_path"),
             py::arg("decision_callback"),
             py::arg("event_callback") = py::none(),
             py::arg("backtrack_limit") = 97, py::arg("seed") = 14);
  module.def("score_actor", &score_actor, py::arg("actor_path"),
             py::arg("mode"), py::arg("objective"), py::arg("candidates"));
}
