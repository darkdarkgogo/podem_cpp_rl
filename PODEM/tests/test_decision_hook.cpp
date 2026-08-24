#include "atpg.h"

#include <cassert>
#include <memory>

class CountingPolicy : public smartatpg::DecisionPolicy {
public:
  int select(const smartatpg::DecisionRequest &request) override {
    assert(request.candidate_names.size() > 1);
    if (request.mode == smartatpg::DecisionMode::BACKTRACE) {
      ++backtrace_decisions;
      assert(!request.objective_name.empty());
    } else {
      ++propagation_decisions;
    }
    return 0;
  }

  void on_episode_start(const std::string &fault_id) override {
    assert(!fault_id.empty());
    ++episode_starts;
  }

  void on_episode_end(const smartatpg::EpisodeResult &result) override {
    assert(!result.fault_id.empty());
    ++episode_ends;
  }

  unsigned long episode_starts = 0;
  unsigned long episode_ends = 0;
  unsigned long backtrace_decisions = 0;
  unsigned long propagation_decisions = 0;
};

int main(int argc, char **argv) {
  assert(argc == 2);
  std::shared_ptr<CountingPolicy> policy = std::make_shared<CountingPolicy>();
  ATPG atpg;
  atpg.detected_num = 1;
  atpg.set_decision_policy(policy);
  atpg.input(argv[1]);
  atpg.level_circuit();
  atpg.rearrange_gate_inputs();
  atpg.create_dummy_gate();
  atpg.generate_fault_list();
  atpg.test();

  assert(policy->episode_starts > 0);
  assert(policy->episode_starts == policy->episode_ends);
  assert(policy->backtrace_decisions > 0);
  return 0;
}
