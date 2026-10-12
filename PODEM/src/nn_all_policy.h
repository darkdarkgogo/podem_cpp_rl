#ifndef NN_ALL_PODEM_POLICY_H
#define NN_ALL_PODEM_POLICY_H

#include "rl_policy.h"

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace smartatpg {

class NnAllModel {
public:
  explicit NnAllModel(const std::string &path);
  float score(const std::array<float, 11> &features) const;
  const std::string &manifest_hash() const { return manifest_hash_; }

private:
  std::array<float, 11> mean_{};
  std::array<float, 11> scale_{};
  std::array<float, 660> w1_{};
  std::array<float, 60> b1_{};
  std::array<float, 60> w2_{};
  float b2_ = 0;
  std::string manifest_hash_;
};

class NnAllPolicy : public DecisionPolicy {
public:
  NnAllPolicy(const std::string &model_path, const std::string &features_path,
              const std::string &expected_circuit_hash,
              const std::vector<std::string> &gate_names_by_id);
  int select(const DecisionRequest &request) override;
  bool needs_gate_names() const override { return false; }
  bool supports(DecisionMode mode) const override {
    return mode == DecisionMode::BACKTRACE;
  }
  const std::string &manifest_hash() const { return model_.manifest_hash(); }

private:
  struct Gate {
    int type = -1;
    float depth = 0;
    float cc0 = 0;
    float cc1 = 0;
    float cop = 0;
  };
  NnAllModel model_;
  std::vector<Gate> gates_;
  std::unordered_map<std::size_t, float> score_cache_;
};

} // namespace smartatpg

#endif
