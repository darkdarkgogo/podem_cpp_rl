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

struct DecisionRequest {
  DecisionMode mode;
  std::string objective_name;
  int objective_value;
  std::vector<std::string> candidate_names;
  unsigned long sequence;
  std::string fault_id;
  int backtracks;
};

struct EpisodeResult {
  std::string fault_id;
  int outcome;
  int backtracks;
  unsigned long decisions;
};

class DecisionPolicy {
public:
  virtual ~DecisionPolicy() {}
  virtual int select(const DecisionRequest &request) = 0;
  virtual void on_episode_start(const std::string &fault_id) {}
  virtual void on_backtrack(unsigned long decision_sequence) {}
  virtual void on_episode_end(const EpisodeResult &result) {}
};

class EmbeddingTable {
public:
  void load(const std::string &path, const std::string &expected_circuit_hash);
  const std::vector<float> &at(const std::string &name) const;
  std::size_t dimension() const { return dimension_; }
  std::size_t size() const { return embeddings_.size(); }

private:
  std::size_t dimension_ = 0;
  std::unordered_map<std::string, std::vector<float> > embeddings_;
};

class ActorModel {
public:
  void load(const std::string &path);
  std::vector<float> backtrace_logits(
      const std::vector<float> &objective,
      const std::vector<std::vector<float> > &candidates) const;
  std::vector<float> propagation_logits(
      const std::vector<std::vector<float> > &candidates) const;
  std::size_t embedding_dimension() const { return embedding_dim_; }

private:
  struct Tensor {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<float> values;
  };

  std::vector<float> encode_gate(const std::vector<float> &embedding,
                                 std::size_t mode) const;
  std::vector<float> dense(const Tensor &weight, const Tensor &bias,
                           const std::vector<float> &input,
                           bool apply_tanh) const;
  const Tensor &tensor(const std::string &name) const;

  std::size_t embedding_dim_ = 0;
  std::size_t hidden_dim_ = 0;
  std::unordered_map<std::string, Tensor> tensors_;
};

class NativeActorPolicy : public DecisionPolicy {
public:
  NativeActorPolicy(const std::string &embedding_path,
                    const std::string &actor_path,
                    const std::string &expected_circuit_hash);
  int select(const DecisionRequest &request) override;

private:
  EmbeddingTable embeddings_;
  ActorModel actor_;
};

std::string fnv1a_file_hash(const std::string &path);
std::string decision_mode_name(DecisionMode mode);

} // namespace smartatpg

#endif
