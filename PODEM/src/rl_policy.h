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
  bool action_mask[2] = {true, true};
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
  const std::string &backend() const { return backend_; }
  const std::string &schema() const { return schema_; }
  const std::string &graph_config() const { return graph_config_; }
  const std::string &encoder_variant() const { return encoder_variant_; }
  std::size_t actor_input_dimension() const { return actor_input_dim_; }
  std::size_t action_mask_dimension() const { return action_mask_dim_; }
  std::size_t decision_state_dimension() const { return decision_state_dim_; }
  std::size_t policy_state_dimension() const { return policy_state_dim_; }
  const std::string &snapshot() const { return snapshot_; }
  void clear();

private:
  std::size_t dimension_ = 0;
  std::size_t actor_input_dim_ = 0;
  std::size_t action_mask_dim_ = 0;
  std::size_t decision_state_dim_ = 0;
  std::size_t policy_state_dim_ = 0;
  std::string backend_ = "smartatpg", schema_, encoder_variant_, graph_config_, snapshot_;
  std::unordered_map<std::string, std::vector<float> > embeddings_;
};

class ActorModel {
public:
  ActorModel() = default;
  ActorModel(const ActorModel &) = delete;
  ActorModel &operator=(const ActorModel &) = delete;

  void load(const std::string &path);
  std::vector<float> backtrace_action_logits(
      const std::vector<float> &objective, int objective_value) const;
  std::size_t embedding_dimension() const { return embedding_dim_; }
  std::size_t gate_embedding_dimension() const { return gate_embedding_dim_; }
  std::size_t hidden_dimension() const { return hidden_dim_; }
  const std::string &backend() const { return backend_; }
  const std::string &schema() const { return schema_; }
  const std::string &graph_config() const { return graph_config_; }
  const std::string &encoder_variant() const { return encoder_variant_; }
  std::size_t action_mask_dimension() const { return action_mask_dim_; }
  std::size_t decision_state_dimension() const { return decision_state_dim_; }
  const std::string &snapshot() const { return snapshot_; }

private:
  struct Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> values;
  };

  void backtrace_action_logits_into(const float *objective,
                                    int objective_value, float *state,
                                    float *hidden, float *logits) const;
  const Tensor &tensor(const std::string &name) const;

  std::size_t embedding_dim_ = 0;
  std::size_t gate_embedding_dim_ = 0;
  std::size_t action_mask_dim_ = 0;
  std::size_t decision_state_dim_ = 0;
  std::size_t hidden_dim_ = 0;
  int version_ = 0;
  std::string backend_ = "smartatpg", schema_, encoder_variant_, graph_config_, snapshot_;
  std::unordered_map<std::string, Tensor> tensors_;
  const Tensor *gate_weight_ = nullptr;
  const Tensor *gate_bias_ = nullptr;
  const Tensor *objective_value_embedding_ = nullptr;

  friend class NativeActorPolicy;
};

class NativeActorPolicy : public DecisionPolicy {
public:
  NativeActorPolicy(const std::string &embedding_path,
                    const std::string &actor_path,
                    const std::string &expected_circuit_hash,
                    const std::vector<std::string> &gate_names_by_id,
                    const std::string &expected_backend = "");
  NativeActorPolicy(const NativeActorPolicy &) = delete;
  NativeActorPolicy &operator=(const NativeActorPolicy &) = delete;
  int select(const DecisionRequest &request) override;
  bool needs_gate_names() const override { return false; }
  bool supports(DecisionMode mode) const override {
    return mode == DecisionMode::BACKTRACE;
  }

private:
  ActorModel actor_;
  std::size_t gate_count_ = 0;
  std::vector<float> state_buffer_;
  std::vector<float> hidden_buffer_;
  std::vector<float> v2_embedding_cache_;
  std::vector<float> v2_policy_input_buffer_;
  std::vector<float> v2_logits_cache_;
  std::vector<unsigned char> v2_cache_valid_;
  std::size_t v2_variants_per_gate_ = 0;
  bool v2_mask_is_actor_input_ = false;
};

std::string fnv1a_file_hash(const std::string &path);
std::string decision_mode_name(DecisionMode mode);
RlMode parse_rl_mode(const std::string &value);
std::string rl_mode_name(RlMode mode);
bool rl_mode_enables(RlMode mode, DecisionMode decision);

} // namespace smartatpg

#endif
