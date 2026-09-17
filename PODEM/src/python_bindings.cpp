#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "atpg.h"
#include "rl_policy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

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

struct NativeValidationRecord {
  std::string fault_id;
  int outcome;
  int backtracks;
  unsigned long backtrace_steps;
  double reward;
  double seconds;
};

std::string json_string(const std::string &value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "\\u00" << std::hex << std::setw(2) << std::setfill('0')
                 << static_cast<int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  output << '"';
  return output.str();
}

class NativeValidationPolicy : public smartatpg::DecisionPolicy {
public:
  explicit NativeValidationPolicy(
      std::shared_ptr<smartatpg::DecisionPolicy> actor, std::size_t total_faults,
      int seed, const std::string &journal_path,
      const std::string &circuit_name)
      : actor_(std::move(actor)), total_faults_(total_faults), seed_(seed),
        circuit_name_(circuit_name) {
    if (!journal_path.empty()) {
      if (circuit_name_.empty()) {
        throw std::invalid_argument(
            "Native validation journal requires a circuit name");
      }
      journal_.open(journal_path, std::ios::out | std::ios::app | std::ios::binary);
      if (!journal_.good()) {
        throw std::runtime_error(
            "Cannot open native validation journal: " + journal_path);
      }
    }
  }

  void start_run() {
    interval_started_ = std::chrono::steady_clock::now();
    run_started_ = true;
  }

  bool needs_gate_names() const override { return false; }
  bool wants_training_events() const override { return true; }
  bool supports(smartatpg::DecisionMode mode) const override {
    return actor_->supports(mode);
  }
  smartatpg::DecisionTimingStats timing_stats() const override {
    return actor_->timing_stats();
  }
  int select(const smartatpg::DecisionRequest &request) override {
    const int action = actor_->select(request);
    decision_sequences_.insert(request.sequence);
    return action;
  }
  void on_episode_start(const std::string &fault_id) override {
    std::srand(seed_);
    current_fault_id_ = fault_id;
    decision_sequences_.clear();
    reward_ = 0.0;
  }
  void on_backtrace_step(unsigned long sequence) override {
    if (decision_sequences_.count(sequence)) reward_ -= 0.1;
  }
  void on_pi_not_done(unsigned long sequence, int backtracks,
                      unsigned long pi_visits) override {
    if (decision_sequences_.count(sequence)) {
      reward_ += 10.0 - 7.5 * std::exp(0.07 * (backtracks + pi_visits));
    }
  }
  void on_episode_end(const smartatpg::EpisodeResult &result) override {
    if (result.fault_id != current_fault_id_) {
      throw std::runtime_error("Native validation fault ID changed mid-episode");
    }
    reward_ += result.outcome == TRUE ? 100.0 : -100.0;
    records_.push_back({result.fault_id, result.outcome, result.backtracks,
                        result.backtrace_steps, reward_, 0.0});
  }
  void on_episode_complete() override {
    if (!run_started_ || records_.empty()) {
      throw std::runtime_error("Native validation completion is out of order");
    }
    const auto completed = std::chrono::steady_clock::now();
    records_.back().seconds = std::chrono::duration<double>(
        completed - interval_started_).count();
    interval_started_ = completed;
    write_record(records_.back());
    if (records_.size() % 1000 == 0 || records_.size() == total_faults_) {
      if (journal_.is_open()) journal_.flush();
      std::fprintf(stdout, "NATIVE_VALIDATE completed=%zu/%zu\n",
                   records_.size(), total_faults_);
      std::fflush(stdout);
    }
  }
  const std::vector<NativeValidationRecord> &records() const { return records_; }

private:
  void write_record(const NativeValidationRecord &record) {
    if (!journal_.is_open()) return;
    const int detected = record.outcome == TRUE ? 1 : 0;
    const int redundant = record.outcome == FALSE ? 1 : 0;
    const int aborted = record.outcome != TRUE && record.outcome != FALSE ? 1 : 0;
    journal_ << "{\"aborted\":" << aborted
             << ",\"atpg_seconds\":" << std::setprecision(17) << record.seconds
             << ",\"backtrace_steps\":" << record.backtrace_steps
             << ",\"backtracks\":" << record.backtracks
             << ",\"circuit\":" << json_string(circuit_name_)
             << ",\"detected\":" << detected
             << ",\"fault_id\":" << json_string(record.fault_id)
             << ",\"outcome\":" << record.outcome
             << ",\"redundant\":" << redundant
             << ",\"return\":" << std::setprecision(17) << record.reward
             << ",\"test_vectors\":" << detected << "}\n";
    if (!journal_.good()) {
      throw std::runtime_error("Cannot write native validation journal");
    }
  }

  std::shared_ptr<smartatpg::DecisionPolicy> actor_;
  std::size_t total_faults_;
  int seed_;
  std::unordered_set<unsigned long> decision_sequences_;
  std::string current_fault_id_;
  std::string circuit_name_;
  double reward_ = 0.0;
  bool run_started_ = false;
  std::chrono::steady_clock::time_point interval_started_;
  std::ofstream journal_;
  std::vector<NativeValidationRecord> records_;
};

py::list run_native_validation(
    const std::string &circuit_path, const std::string &embedding_path,
    const std::string &actor_path, int backtrack_limit, int seed,
    const std::vector<std::string> &fault_ids,
    const std::string &journal_path, const std::string &circuit_name) {
  if (fault_ids.empty()) {
    throw std::invalid_argument("Native validation requires fault IDs");
  }
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_backtrack_limit(backtrack_limit);
  atpg.set_seed(seed);
  atpg.set_total_attempt_num(1);
  atpg.set_SAF_atpg(true);
  atpg.set_SCOAP(true);
  atpg.set_rl_mode("backtrace_rl");
  atpg.set_quiet(true);
  std::shared_ptr<NativeValidationPolicy> policy;
  {
    py::gil_scoped_release release;
    atpg.input(circuit_path);
    atpg.enable_rl_inference(embedding_path, actor_path, "smartatpg");
    policy = std::make_shared<NativeValidationPolicy>(
        atpg.get_decision_policy(), fault_ids.size(), seed,
        journal_path, circuit_name);
    atpg.set_decision_policy(policy);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    atpg.retain_faults(fault_ids);
    atpg.set_drop_detected_faults(false);
    policy->start_run();
    atpg.test();
  }
  const auto &records = policy->records();
  if (records.size() != fault_ids.size()) {
    throw std::runtime_error("Native validation returned the wrong fault count");
  }
  py::list result;
  for (std::size_t index = 0; index < records.size(); ++index) {
    const auto &record = records[index];
    if (record.fault_id != fault_ids[index]) {
      throw std::runtime_error("Native validation returned faults out of order");
    }
    py::dict item;
    item["fault_id"] = record.fault_id;
    item["outcome"] = record.outcome;
    item["backtracks"] = record.backtracks;
    item["backtrace_steps"] = record.backtrace_steps;
    item["return"] = record.reward;
    item["atpg_seconds"] = record.seconds;
    result.append(item);
  }
  return result;
}

py::dict run_stuck_at(const std::string &circuit_path,
                      py::function decision_callback,
                      py::object event_callback, int backtrack_limit,
                      int seed, py::object fault_ids, bool quiet,
                      const std::string &rl_mode,
                      const std::string &fault_map_path, bool use_scoap) {
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
  atpg.set_SCOAP(use_scoap);
  atpg.set_rl_mode(rl_mode);
  atpg.set_decision_policy(policy);
  atpg.set_quiet(quiet);
  atpg.set_fault_map_path(fault_map_path);

  double atpg_seconds = 0.0;
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
    const auto atpg_started = std::chrono::steady_clock::now();
    atpg.test();
    atpg_seconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - atpg_started)
                       .count();
    if (!has_fault_filter) {
      atpg.compute_fault_coverage();
    }
  }
  atpg.disable_rl_policy();
  py::dict result = policy->summary();
  result["atpg_seconds"] = atpg_seconds;
  return result;
}

py::list profile_stuck_at(const std::string &circuit_path,
                          int backtrack_limit, int seed,
                          const std::string &fault_map_path, bool use_scoap) {
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_backtrack_limit(backtrack_limit);
  atpg.set_seed(seed);
  atpg.set_total_attempt_num(1);
  atpg.set_SAF_atpg(true);
  atpg.set_SCOAP(use_scoap);
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
             py::arg("fault_map_path") = "", py::arg("use_scoap") = false);
  module.def("run_native_validation", &run_native_validation,
             py::arg("circuit_path"), py::arg("embedding_path"),
             py::arg("actor_path"), py::arg("backtrack_limit"),
             py::arg("seed"), py::arg("fault_ids"),
             py::arg("journal_path") = "", py::arg("circuit_name") = "");
  module.def("profile_stuck_at", &profile_stuck_at,
             py::arg("circuit_path"), py::arg("backtrack_limit") = 97,
             py::arg("seed") = 14, py::arg("fault_map_path") = "",
             py::arg("use_scoap") = false);
  module.def("catalog_stuck_at", &catalog_stuck_at,
             py::arg("circuit_path"), py::arg("fault_map_path") = "");
  module.def("score_actor_v2", &score_actor_v2, py::arg("actor_path"),
             py::arg("objective"), py::arg("objective_value"));
  module.def("validate_actor_artifacts", &validate_actor_artifacts,
             py::arg("embedding_path"), py::arg("actor_path"), py::arg("circuit_hash"),
             py::arg("names"), py::arg("backend") = "");
}
