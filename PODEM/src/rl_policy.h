#ifndef SMARTATPG_RL_POLICY_H
#define SMARTATPG_RL_POLICY_H

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace smartatpg {

enum class DecisionMode {
  BACKTRACE,
  PROPAGATION,
};

enum class RlMode {
  BACKTRACE_RL,
  PROPAGATE_RL,
  BOTH_RL,
};

struct DecisionRequest {
  DecisionMode mode;
  std::string objective_name;
  std::size_t objective_id = static_cast<std::size_t>(-1);
  int objective_value;
  std::vector<std::string> candidate_names;
  const std::size_t *candidate_ids = nullptr;
  std::size_t candidate_count = 0;
  int heuristic_action = -1;
  unsigned long sequence;
  std::string fault_id;
  int backtracks;
};

struct EpisodeResult {
  std::string fault_id;
  int outcome;
  int backtracks;
  unsigned long backtrace_steps;
  unsigned long pi_visits;
  unsigned long decisions;
};

class DecisionPolicy {
public:
  virtual ~DecisionPolicy() {}
  virtual int select(const DecisionRequest &request) = 0;
  virtual bool needs_gate_names() const { return true; }
  virtual bool wants_training_events() const { return false; }
  virtual bool supports(DecisionMode mode) const { return true; }
  virtual void on_episode_start(const std::string &fault_id) {}
  virtual void on_backtrack(unsigned long decision_sequence) {}
  virtual void on_backtrace_step(unsigned long decision_sequence) {}
  virtual void on_pi_not_done(unsigned long decision_sequence, int backtracks,
                              unsigned long pi_visits) {}
  virtual void on_episode_end(const EpisodeResult &result) {}
};

class EmbeddingTable {
public:
  void load(const std::string &path, const std::string &expected_circuit_hash);
  const std::vector<float> &at(const std::string &name) const;
  std::size_t dimension() const { return dimension_; }
  std::size_t size() const { return embeddings_.size(); }
  void clear();

private:
  std::size_t dimension_ = 0;
  std::unordered_map<std::string, std::vector<float> > embeddings_;
};

class ActorModel {
public:
  ActorModel() = default;
  ActorModel(const ActorModel &) = delete;
  ActorModel &operator=(const ActorModel &) = delete;

  void load(const std::string &path);
  std::vector<float> backtrace_logits(
      const std::vector<float> &objective,
      const std::vector<std::vector<float> > &candidates) const;
  std::vector<float> propagation_logits(
      const std::vector<std::vector<float> > &candidates) const;
  std::vector<float> optimized_logits(
      DecisionMode mode, const std::vector<float> &objective,
      const std::vector<std::vector<float> > &candidates) const;
  std::vector<float> backtrace_action_logits(
      const std::vector<float> &objective, int objective_value) const;
  std::size_t embedding_dimension() const { return embedding_dim_; }
  std::size_t hidden_dimension() const { return hidden_dim_; }
  bool is_v2() const { return version_ == 2; }

private:
  struct Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> values;
  };

  struct ActorHead {
    const Tensor *hidden_weight = nullptr;
    const Tensor *hidden_bias = nullptr;
    const Tensor *output_weight = nullptr;
    const Tensor *output_bias = nullptr;
  };

  std::vector<float> encode_gate(const std::vector<float> &embedding,
                                 std::size_t mode) const;
  void encode_gate_into(const std::vector<float> &embedding, std::size_t mode,
                        float *output) const;
  float score_encoded(DecisionMode mode, const float *state,
                      const float *candidate,
                      std::vector<float> &hidden_buffer) const;
  const ActorHead &head(DecisionMode mode) const;
  std::vector<float> dense(const Tensor &weight, const Tensor &bias,
                           const std::vector<float> &input,
                           bool apply_tanh) const;
  void backtrace_action_logits_into(const float *objective,
                                    int objective_value, float *state,
                                    float *hidden, float *logits) const;
  const Tensor &tensor(const std::string &name) const;

  std::size_t embedding_dim_ = 0;
  std::size_t hidden_dim_ = 0;
  int version_ = 0;
  std::unordered_map<std::string, Tensor> tensors_;
  const Tensor *gate_weight_ = nullptr;
  const Tensor *gate_bias_ = nullptr;
  const Tensor *mode_embedding_ = nullptr;
  const Tensor *objective_value_embedding_ = nullptr;
  ActorHead backtrace_head_;
  ActorHead propagation_head_;

  friend class NativeActorPolicy;
};

class NativeActorPolicy : public DecisionPolicy {
public:
  NativeActorPolicy(const std::string &embedding_path,
                    const std::string &actor_path,
                    const std::string &expected_circuit_hash,
                    const std::vector<std::string> &gate_names_by_id);
  NativeActorPolicy(const NativeActorPolicy &) = delete;
  NativeActorPolicy &operator=(const NativeActorPolicy &) = delete;
  int select(const DecisionRequest &request) override;
  bool needs_gate_names() const override { return false; }
  bool supports(DecisionMode mode) const override {
    return !actor_.is_v2() || mode == DecisionMode::BACKTRACE;
  }

private:
  const float *cached_gate(DecisionMode mode, std::size_t gate_id) const;

  ActorModel actor_;
  std::size_t gate_count_ = 0;
  std::vector<float> backtrace_cache_;
  std::vector<float> propagation_cache_;
  std::vector<float> state_buffer_;
  std::vector<float> hidden_buffer_;
  std::vector<float> v2_embedding_cache_;
  std::vector<float> v2_logits_cache_;
  std::vector<unsigned char> v2_cache_valid_;
};

std::string fnv1a_file_hash(const std::string &path);
std::string decision_mode_name(DecisionMode mode);
RlMode parse_rl_mode(const std::string &value);
std::string rl_mode_name(RlMode mode);
bool rl_mode_enables(RlMode mode, DecisionMode decision);

} // namespace smartatpg

#endif
