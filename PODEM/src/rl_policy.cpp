#include "rl_policy.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace smartatpg {
namespace {

void require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_finite(float value, const std::string &context) {
  require(std::isfinite(value), "Non-finite value in " + context);
}

std::vector<std::string> required_tensor_names() {
  return {
      "gate_encoder.0.weight",
      "gate_encoder.0.bias",
      "mode_embedding.weight",
      "backtrace_actor.0.weight",
      "backtrace_actor.0.bias",
      "backtrace_actor.2.weight",
      "backtrace_actor.2.bias",
      "propagation_actor.0.weight",
      "propagation_actor.0.bias",
      "propagation_actor.2.weight",
      "propagation_actor.2.bias",
  };
}

} // namespace

void EmbeddingTable::load(const std::string &path,
                          const std::string &expected_circuit_hash) {
  std::ifstream input(path.c_str());
  require(input.good(), "Cannot open embedding file: " + path);

  std::string header;
  std::getline(input, header);
  require(header == "SMARTATPG_EMBEDDINGS_V1",
          "Unsupported embedding format in: " + path);

  std::string key;
  std::string circuit_hash;
  std::size_t expected_count = 0;
  require(static_cast<bool>(input >> key >> circuit_hash) && key == "circuit_hash",
          "Missing circuit_hash in embedding file: " + path);
  require(circuit_hash == expected_circuit_hash,
          "Embedding circuit hash mismatch: expected " + expected_circuit_hash +
              ", found " + circuit_hash);
  require(static_cast<bool>(input >> key >> dimension_) && key == "dimension" &&
              dimension_ > 0,
          "Invalid embedding dimension in: " + path);
  require(static_cast<bool>(input >> key >> expected_count) && key == "count",
          "Invalid embedding count in: " + path);

  embeddings_.clear();
  for (std::size_t row = 0; row < expected_count; ++row) {
    std::string name;
    require(static_cast<bool>(input >> name),
            "Missing embedding name at row " + std::to_string(row));
    std::vector<float> values(dimension_);
    for (std::size_t col = 0; col < dimension_; ++col) {
      require(static_cast<bool>(input >> values[col]),
              "Truncated embedding for gate: " + name);
      require_finite(values[col], "embedding for gate " + name);
    }
    require(embeddings_.insert(std::make_pair(name, values)).second,
            "Duplicate embedding name: " + name);
  }
  require(embeddings_.size() == expected_count,
          "Embedding count does not match file header");
  std::string trailing;
  require(!(input >> trailing), "Unexpected trailing data in embedding file: " + path);
}

const std::vector<float> &EmbeddingTable::at(const std::string &name) const {
  const auto found = embeddings_.find(name);
  if (found == embeddings_.end()) {
    throw std::runtime_error("Missing DeepGate embedding for wire: " + name);
  }
  return found->second;
}

void ActorModel::load(const std::string &path) {
  std::ifstream input(path.c_str());
  require(input.good(), "Cannot open actor file: " + path);

  std::string header;
  std::getline(input, header);
  require(header == "SMARTATPG_ACTOR_V1",
          "Unsupported actor format in: " + path);

  std::string key;
  require(static_cast<bool>(input >> key >> embedding_dim_) &&
              key == "embedding_dim" && embedding_dim_ > 0,
          "Invalid actor embedding dimension in: " + path);
  require(static_cast<bool>(input >> key >> hidden_dim_) && key == "hidden_dim" &&
              hidden_dim_ > 0,
          "Invalid actor hidden dimension in: " + path);

  tensors_.clear();
  bool found_end = false;
  while (input >> key) {
    if (key == "end") {
      found_end = true;
      break;
    }
    require(key == "tensor", "Expected tensor entry in actor file: " + path);
    std::string name;
    Tensor value;
    require(static_cast<bool>(input >> name >> value.rows >> value.cols) &&
                value.rows > 0 && value.cols > 0,
            "Invalid tensor header in actor file: " + path);
    value.values.resize(value.rows * value.cols);
    for (std::size_t i = 0; i < value.values.size(); ++i) {
      require(static_cast<bool>(input >> value.values[i]),
              "Truncated actor tensor: " + name);
      require_finite(value.values[i], "actor tensor " + name);
    }
    require(tensors_.insert(std::make_pair(name, value)).second,
            "Duplicate actor tensor: " + name);
  }
  require(found_end, "Actor file is missing the end marker: " + path);

  const std::vector<std::string> names = required_tensor_names();
  for (const std::string &name : names) {
    tensor(name);
  }

  require(tensor("gate_encoder.0.weight").rows == hidden_dim_ &&
              tensor("gate_encoder.0.weight").cols == embedding_dim_,
          "gate_encoder weight dimensions do not match actor header");
  require(tensor("gate_encoder.0.bias").rows *
                  tensor("gate_encoder.0.bias").cols ==
              hidden_dim_,
          "gate_encoder bias dimensions do not match actor header");
  require(tensor("mode_embedding.weight").rows == 2 &&
              tensor("mode_embedding.weight").cols == hidden_dim_,
          "mode_embedding dimensions must be [2, hidden_dim]");

  const auto validate_actor = [this](const std::string &prefix) {
    const Tensor &hidden_weight = tensor(prefix + ".0.weight");
    const Tensor &hidden_bias = tensor(prefix + ".0.bias");
    const Tensor &output_weight = tensor(prefix + ".2.weight");
    const Tensor &output_bias = tensor(prefix + ".2.bias");
    require(hidden_weight.rows == hidden_dim_ &&
                hidden_weight.cols == hidden_dim_ * 2,
            prefix + " hidden weight dimensions are invalid");
    require(hidden_bias.values.size() == hidden_dim_,
            prefix + " hidden bias dimensions are invalid");
    require(output_weight.rows == 1 && output_weight.cols == hidden_dim_,
            prefix + " output weight dimensions are invalid");
    require(output_bias.values.size() == 1,
            prefix + " output bias dimensions are invalid");
  };
  validate_actor("backtrace_actor");
  validate_actor("propagation_actor");
}

const ActorModel::Tensor &ActorModel::tensor(const std::string &name) const {
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    throw std::runtime_error("Missing actor tensor: " + name);
  }
  return found->second;
}

std::vector<float> ActorModel::dense(const Tensor &weight, const Tensor &bias,
                                     const std::vector<float> &input,
                                     bool apply_tanh) const {
  require(weight.cols == input.size(), "Dense input dimension mismatch");
  require(bias.values.size() == weight.rows, "Dense bias dimension mismatch");
  std::vector<float> output(weight.rows, 0.0f);
  for (std::size_t row = 0; row < weight.rows; ++row) {
    float value = bias.values[row];
    const std::size_t offset = row * weight.cols;
    for (std::size_t col = 0; col < weight.cols; ++col) {
      value += weight.values[offset + col] * input[col];
    }
    output[row] = apply_tanh ? std::tanh(value) : value;
  }
  return output;
}

std::vector<float> ActorModel::encode_gate(
    const std::vector<float> &embedding, std::size_t mode) const {
  require(embedding.size() == embedding_dim_,
          "Gate embedding dimension does not match actor");
  require(mode < 2, "Actor mode index is out of range");
  std::vector<float> encoded =
      dense(tensor("gate_encoder.0.weight"), tensor("gate_encoder.0.bias"),
            embedding, true);
  const Tensor &mode_embedding = tensor("mode_embedding.weight");
  for (std::size_t i = 0; i < hidden_dim_; ++i) {
    encoded[i] += mode_embedding.values[mode * hidden_dim_ + i];
  }
  return encoded;
}

std::vector<float> ActorModel::backtrace_logits(
    const std::vector<float> &objective,
    const std::vector<std::vector<float> > &candidates) const {
  require(!candidates.empty(), "Backtrace actor requires at least one candidate");
  const std::vector<float> objective_repr = encode_gate(objective, 0);
  std::vector<float> logits;
  logits.reserve(candidates.size());
  for (const std::vector<float> &candidate : candidates) {
    const std::vector<float> candidate_repr = encode_gate(candidate, 0);
    std::vector<float> pair = objective_repr;
    pair.insert(pair.end(), candidate_repr.begin(), candidate_repr.end());
    const std::vector<float> hidden =
        dense(tensor("backtrace_actor.0.weight"),
              tensor("backtrace_actor.0.bias"), pair, true);
    logits.push_back(dense(tensor("backtrace_actor.2.weight"),
                           tensor("backtrace_actor.2.bias"), hidden, false)[0]);
  }
  return logits;
}

std::vector<float> ActorModel::propagation_logits(
    const std::vector<std::vector<float> > &candidates) const {
  require(!candidates.empty(), "Propagation actor requires at least one candidate");
  std::vector<std::vector<float> > encoded;
  std::vector<float> state(hidden_dim_, 0.0f);
  for (const std::vector<float> &candidate : candidates) {
    encoded.push_back(encode_gate(candidate, 1));
    for (std::size_t i = 0; i < hidden_dim_; ++i) {
      state[i] += encoded.back()[i];
    }
  }
  for (float &value : state) {
    value /= static_cast<float>(encoded.size());
  }

  std::vector<float> logits;
  logits.reserve(encoded.size());
  for (const std::vector<float> &candidate_repr : encoded) {
    std::vector<float> pair = state;
    pair.insert(pair.end(), candidate_repr.begin(), candidate_repr.end());
    const std::vector<float> hidden =
        dense(tensor("propagation_actor.0.weight"),
              tensor("propagation_actor.0.bias"), pair, true);
    logits.push_back(dense(tensor("propagation_actor.2.weight"),
                           tensor("propagation_actor.2.bias"), hidden, false)[0]);
  }
  return logits;
}

NativeActorPolicy::NativeActorPolicy(const std::string &embedding_path,
                                     const std::string &actor_path,
                                     const std::string &expected_circuit_hash) {
  embeddings_.load(embedding_path, expected_circuit_hash);
  actor_.load(actor_path);
  require(embeddings_.dimension() == actor_.embedding_dimension(),
          "Embedding and actor dimensions do not match");
}

int NativeActorPolicy::select(const DecisionRequest &request) {
  require(request.candidate_names.size() > 1,
          "Policy should only be called for multi-candidate decisions");
  std::vector<std::vector<float> > candidates;
  candidates.reserve(request.candidate_names.size());
  for (const std::string &name : request.candidate_names) {
    candidates.push_back(embeddings_.at(name));
  }

  std::vector<float> logits;
  if (request.mode == DecisionMode::BACKTRACE) {
    logits = actor_.backtrace_logits(embeddings_.at(request.objective_name),
                                     candidates);
  } else {
    logits = actor_.propagation_logits(candidates);
  }
  require(logits.size() == request.candidate_names.size(),
          "Actor returned an unexpected number of logits");

  std::size_t best = 0;
  for (std::size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > logits[best]) {
      best = i;
    }
  }
  return static_cast<int>(best);
}

std::string fnv1a_file_hash(const std::string &path) {
  std::ifstream input(path.c_str(), std::ios::binary);
  require(input.good(), "Cannot hash circuit file: " + path);
  std::uint64_t hash = UINT64_C(14695981039346656037);
  char buffer[8192];
  while (input.good()) {
    input.read(buffer, sizeof(buffer));
    const std::streamsize count = input.gcount();
    for (std::streamsize i = 0; i < count; ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= UINT64_C(1099511628211);
    }
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0') << std::setw(16) << hash;
  return text.str();
}

std::string decision_mode_name(DecisionMode mode) {
  return mode == DecisionMode::BACKTRACE ? "backtrace" : "propagation";
}

} // namespace smartatpg
