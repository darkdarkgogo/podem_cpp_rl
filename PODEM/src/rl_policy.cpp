#include "rl_policy.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

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

std::ifstream open_artifact(const std::string &path, std::ios::openmode mode = std::ios::in) {
#ifdef _WIN32
  const UINT codepage = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
      path.c_str(), -1, nullptr, 0) > 0 ? CP_UTF8 : CP_ACP;
  const int count = MultiByteToWideChar(codepage, 0, path.c_str(), -1, nullptr, 0);
  require(count > 0, "Invalid artifact path encoding");
  std::vector<wchar_t> wide(static_cast<std::size_t>(count));
  require(MultiByteToWideChar(codepage, 0, path.c_str(), -1, wide.data(), count) > 0,
          "Cannot decode artifact path");
  return std::ifstream(wide.data(), mode);
#else
  return std::ifstream(path.c_str(), mode);
#endif
}

std::vector<std::string> required_tensor_names(int version) {
  if (version >= 2) {
    return {
        "gate_encoder.0.weight",
        "gate_encoder.0.bias",
        "objective_value_embedding.weight",
        "backtrace_actor.0.weight",
        "backtrace_actor.0.bias",
        "backtrace_actor.2.weight",
        "backtrace_actor.2.bias",
    };
  }
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

void read_backend_metadata(std::istream &input, std::string &backend,
                           std::string &schema, std::string &snapshot) {
  std::string key;
  require(static_cast<bool>(input >> key >> backend) && key == "backend" &&
              backend == "smartatpg", "Invalid artifact backend");
  require(static_cast<bool>(input >> key >> schema) && key == "feature_schema" &&
              schema == "SMARTATPG_FEATURES_V1", "Invalid artifact feature schema");
  require(static_cast<bool>(input >> key >> snapshot) && key == "snapshot" &&
              snapshot.size() == 64 &&
              snapshot.find_first_not_of("0123456789abcdef") == std::string::npos,
          "Invalid artifact snapshot identifier");
}

} // namespace

void EmbeddingTable::load(const std::string &path,
                          const std::string &expected_circuit_hash) {
  std::ifstream input = open_artifact(path);
  require(input.good(), "Cannot open embedding file: " + path);

  std::string header;
  std::getline(input, header);
  require(header == "SMARTATPG_EMBEDDINGS_V1" || header == "SMARTATPG_EMBEDDINGS_V2",
          "Unsupported embedding format in: " + path);
  backend_ = "deepgate";
  schema_.clear();
  snapshot_.clear();
  if (header == "SMARTATPG_EMBEDDINGS_V2") {
    read_backend_metadata(input, backend_, schema_, snapshot_);
  }

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
  require(backend_ != "smartatpg" || dimension_ == 80,
          "SmartATPG descriptor dimension must be 80");
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
    require(backend_ != "smartatpg" || (values[78] == 1.0f && values[79] == 1.0f),
            "SmartATPG native descriptors require mask [1,1]");
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
    throw std::runtime_error("Missing policy descriptor for wire: " + name);
  }
  return found->second;
}

void EmbeddingTable::clear() {
  embeddings_.clear();
  embeddings_.rehash(0);
  dimension_ = 0;
}

void ActorModel::load(const std::string &path) {
  std::ifstream input = open_artifact(path);
  require(input.good(), "Cannot open actor file: " + path);

  std::string header;
  std::getline(input, header);
  version_ = header == "SMARTATPG_ACTOR_V1"
                 ? 1
                 : (header == "SMARTATPG_ACTOR_V2" ? 2 :
                    (header == "SMARTATPG_ACTOR_V3" ? 3 : 0));
  require(version_ != 0,
          "Unsupported actor format in: " + path);
  backend_ = "deepgate";
  schema_.clear();
  snapshot_.clear();
  if (version_ == 3) {
    read_backend_metadata(input, backend_, schema_, snapshot_);
  }

  std::string key;
  require(static_cast<bool>(input >> key >> embedding_dim_) &&
              key == "embedding_dim" && embedding_dim_ > 0,
          "Invalid actor embedding dimension in: " + path);
  require(static_cast<bool>(input >> key >> hidden_dim_) && key == "hidden_dim" &&
              hidden_dim_ > 0,
          "Invalid actor hidden dimension in: " + path);
  require(version_ != 3 || embedding_dim_ == 80,
          "SmartATPG actor descriptor dimension must be 80");

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
  require(!(input >> key), "Unexpected trailing data in actor file: " + path);

  const std::vector<std::string> names = required_tensor_names(version_);
  require(tensors_.size() == names.size(), "Unexpected actor tensor count");
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
  const auto validate_v1_actor = [this](const std::string &prefix) {
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

  gate_weight_ = &tensor("gate_encoder.0.weight");
  gate_bias_ = &tensor("gate_encoder.0.bias");
  mode_embedding_ = nullptr;
  objective_value_embedding_ = nullptr;
  if (is_v2()) {
    const Tensor &value_embedding = tensor("objective_value_embedding.weight");
    require(value_embedding.rows == 2 && value_embedding.cols == hidden_dim_,
            "objective_value_embedding dimensions must be [2, hidden_dim]");
    require(tensor("backtrace_actor.0.weight").rows == hidden_dim_ &&
                tensor("backtrace_actor.0.weight").cols == hidden_dim_,
            "V2 backtrace hidden weight dimensions are invalid");
    require(tensor("backtrace_actor.0.bias").values.size() == hidden_dim_,
            "V2 backtrace hidden bias dimensions are invalid");
    require(tensor("backtrace_actor.2.weight").rows == 2 &&
                tensor("backtrace_actor.2.weight").cols == hidden_dim_,
            "V2 backtrace output weight dimensions must be [2, hidden_dim]");
    require(tensor("backtrace_actor.2.bias").values.size() == 2,
            "V2 backtrace output bias dimensions must be [2]");
    objective_value_embedding_ = &value_embedding;
    return;
  }

  require(tensor("mode_embedding.weight").rows == 2 &&
              tensor("mode_embedding.weight").cols == hidden_dim_,
          "mode_embedding dimensions must be [2, hidden_dim]");
  validate_v1_actor("backtrace_actor");
  validate_v1_actor("propagation_actor");
  mode_embedding_ = &tensor("mode_embedding.weight");
  backtrace_head_.hidden_weight = &tensor("backtrace_actor.0.weight");
  backtrace_head_.hidden_bias = &tensor("backtrace_actor.0.bias");
  backtrace_head_.output_weight = &tensor("backtrace_actor.2.weight");
  backtrace_head_.output_bias = &tensor("backtrace_actor.2.bias");
  propagation_head_.hidden_weight = &tensor("propagation_actor.0.weight");
  propagation_head_.hidden_bias = &tensor("propagation_actor.0.bias");
  propagation_head_.output_weight = &tensor("propagation_actor.2.weight");
  propagation_head_.output_bias = &tensor("propagation_actor.2.bias");
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
  std::vector<float> encoded(hidden_dim_);
  encode_gate_into(embedding, mode, encoded.data());
  return encoded;
}

void ActorModel::encode_gate_into(const std::vector<float> &embedding,
                                  std::size_t mode, float *output) const {
  require(embedding.size() == embedding_dim_,
          "Gate embedding dimension does not match actor");
  require(mode < 2, "Actor mode index is out of range");
  require(output != nullptr, "Gate encoding output must not be null");
  require(gate_weight_ != nullptr && gate_bias_ != nullptr &&
              mode_embedding_ != nullptr,
          "Actor tensors are not initialized");

  for (std::size_t row = 0; row < hidden_dim_; ++row) {
    float value = gate_bias_->values[row];
    const std::size_t offset = row * embedding_dim_;
    for (std::size_t col = 0; col < embedding_dim_; ++col) {
      value += gate_weight_->values[offset + col] * embedding[col];
    }
    output[row] = std::tanh(value) +
                  mode_embedding_->values[mode * hidden_dim_ + row];
  }
}

const ActorModel::ActorHead &ActorModel::head(DecisionMode mode) const {
  return mode == DecisionMode::BACKTRACE ? backtrace_head_
                                         : propagation_head_;
}

float ActorModel::score_encoded(DecisionMode mode, const float *state,
                                const float *candidate,
                                std::vector<float> &hidden_buffer) const {
  require(state != nullptr && candidate != nullptr,
          "Encoded actor inputs must not be null");
  require(hidden_buffer.size() == hidden_dim_,
          "Actor hidden buffer dimension mismatch");
  const ActorHead &actor_head = head(mode);
  require(actor_head.hidden_weight != nullptr &&
              actor_head.hidden_bias != nullptr &&
              actor_head.output_weight != nullptr &&
              actor_head.output_bias != nullptr,
          "Actor head tensors are not initialized");

  for (std::size_t row = 0; row < hidden_dim_; ++row) {
    float value = actor_head.hidden_bias->values[row];
    const std::size_t offset = row * hidden_dim_ * 2;
    for (std::size_t col = 0; col < hidden_dim_; ++col) {
      value += actor_head.hidden_weight->values[offset + col] * state[col];
    }
    for (std::size_t col = 0; col < hidden_dim_; ++col) {
      value += actor_head.hidden_weight->values[offset + hidden_dim_ + col] *
               candidate[col];
    }
    hidden_buffer[row] = std::tanh(value);
  }

  float score = actor_head.output_bias->values[0];
  for (std::size_t col = 0; col < hidden_dim_; ++col) {
    score += actor_head.output_weight->values[col] * hidden_buffer[col];
  }
  return score;
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

std::vector<float> ActorModel::optimized_logits(
    DecisionMode mode, const std::vector<float> &objective,
    const std::vector<std::vector<float> > &candidates) const {
  require(!candidates.empty(), "Optimized actor requires at least one candidate");
  std::vector<std::vector<float> > encoded;
  encoded.reserve(candidates.size());
  const std::size_t mode_index = mode == DecisionMode::BACKTRACE ? 0 : 1;
  for (const std::vector<float> &candidate : candidates) {
    encoded.push_back(encode_gate(candidate, mode_index));
  }

  std::vector<float> state(hidden_dim_, 0.0f);
  if (mode == DecisionMode::BACKTRACE) {
    state = encode_gate(objective, 0);
  } else {
    for (const std::vector<float> &candidate : encoded) {
      for (std::size_t col = 0; col < hidden_dim_; ++col) {
        state[col] += candidate[col];
      }
    }
    const float divisor = static_cast<float>(encoded.size());
    for (float &value : state) {
      value /= divisor;
    }
  }

  std::vector<float> hidden(hidden_dim_);
  std::vector<float> logits;
  logits.reserve(encoded.size());
  for (const std::vector<float> &candidate : encoded) {
    logits.push_back(
        score_encoded(mode, state.data(), candidate.data(), hidden));
  }
  return logits;
}

std::vector<float> ActorModel::backtrace_action_logits(
    const std::vector<float> &objective, int objective_value) const {
  std::vector<float> state(hidden_dim_);
  std::vector<float> hidden(hidden_dim_);
  std::vector<float> logits(2);
  require(objective.size() == embedding_dim_,
          "Gate embedding dimension does not match V2 actor");
  backtrace_action_logits_into(objective.data(), objective_value, state.data(),
                               hidden.data(), logits.data());
  return logits;
}

void ActorModel::backtrace_action_logits_into(
    const float *objective, int objective_value, float *state, float *hidden,
    float *logits) const {
  require(is_v2(),
          "Object-only backtrace logits require a V2 actor");
  require(objective_value == 0 || objective_value == 1,
          "Backtrace objective value must be 0 or 1");
  require(objective != nullptr && state != nullptr && hidden != nullptr &&
              logits != nullptr,
          "V2 actor buffers must not be null");
  require(gate_weight_ != nullptr && gate_bias_ != nullptr &&
              objective_value_embedding_ != nullptr,
          "V2 actor tensors are not initialized");

  for (std::size_t row = 0; row < hidden_dim_; ++row) {
    float value = gate_bias_->values[row];
    const std::size_t offset = row * embedding_dim_;
    for (std::size_t col = 0; col < embedding_dim_; ++col) {
      value += gate_weight_->values[offset + col] * objective[col];
    }
    state[row] = std::tanh(value) +
                 objective_value_embedding_->values[
                     static_cast<std::size_t>(objective_value) * hidden_dim_ + row];
  }

  const Tensor &hidden_weight = tensor("backtrace_actor.0.weight");
  const Tensor &hidden_bias = tensor("backtrace_actor.0.bias");
  for (std::size_t row = 0; row < hidden_dim_; ++row) {
    float value = hidden_bias.values[row];
    const std::size_t offset = row * hidden_dim_;
    for (std::size_t col = 0; col < hidden_dim_; ++col) {
      value += hidden_weight.values[offset + col] * state[col];
    }
    hidden[row] = std::tanh(value);
  }

  const Tensor &output_weight = tensor("backtrace_actor.2.weight");
  const Tensor &output_bias = tensor("backtrace_actor.2.bias");
  for (std::size_t row = 0; row < 2; ++row) {
    float value = output_bias.values[row];
    const std::size_t offset = row * hidden_dim_;
    for (std::size_t col = 0; col < hidden_dim_; ++col) {
      value += output_weight.values[offset + col] * hidden[col];
    }
    logits[row] = value;
  }
}

NativeActorPolicy::NativeActorPolicy(
    const std::string &embedding_path, const std::string &actor_path,
    const std::string &expected_circuit_hash,
    const std::vector<std::string> &gate_names_by_id,
    const std::string &expected_backend) {
  EmbeddingTable embeddings;
  embeddings.load(embedding_path, expected_circuit_hash);
  actor_.load(actor_path);
  require(expected_backend.empty() || expected_backend == "smartatpg" ||
              expected_backend == "deepgate", "Unknown embedding backend");
  require(expected_backend.empty() || expected_backend == actor_.backend(),
          "Requested embedding backend conflicts with actor backend");
  require(embeddings.backend() == actor_.backend(), "Embedding and actor backend mismatch");
  require(embeddings.schema() == actor_.schema(), "Embedding and actor schema mismatch");
  require(embeddings.snapshot() == actor_.snapshot(), "Embedding and actor snapshot mismatch");
  require(embeddings.dimension() == actor_.embedding_dimension(),
          "Embedding and actor dimensions do not match");
  require(!gate_names_by_id.empty(), "Circuit gate table must not be empty");
  require(std::unordered_set<std::string>(gate_names_by_id.begin(), gate_names_by_id.end()).size() == gate_names_by_id.size(),
          "Circuit gate table contains duplicate wire names");
  require(actor_.backend() != "smartatpg" || embeddings.size() == gate_names_by_id.size(),
          "SmartATPG descriptor count does not match circuit wires");

  gate_count_ = gate_names_by_id.size();
  if (actor_.is_v2()) {
    const std::size_t embedding_dim = actor_.embedding_dimension();
    v2_embedding_cache_.resize(gate_count_ * embedding_dim);
    v2_logits_cache_.resize(gate_count_ * 4);
    v2_cache_valid_.assign(gate_count_ * 2, 0);
    for (std::size_t gate_id = 0; gate_id < gate_count_; ++gate_id) {
      const std::vector<float> &embedding =
          embeddings.at(gate_names_by_id[gate_id]);
      std::copy(embedding.begin(), embedding.end(),
                v2_embedding_cache_.begin() + gate_id * embedding_dim);
    }
    embeddings.clear();
    state_buffer_.resize(actor_.hidden_dimension());
    hidden_buffer_.resize(actor_.hidden_dimension());
    return;
  }

  const std::size_t hidden_dim = actor_.hidden_dimension();
  backtrace_cache_.resize(gate_count_ * hidden_dim);
  propagation_cache_.resize(gate_count_ * hidden_dim);
  for (std::size_t gate_id = 0; gate_id < gate_count_; ++gate_id) {
    const std::vector<float> &embedding = embeddings.at(gate_names_by_id[gate_id]);
    actor_.encode_gate_into(embedding, 0,
                            &backtrace_cache_[gate_id * hidden_dim]);
    actor_.encode_gate_into(embedding, 1,
                            &propagation_cache_[gate_id * hidden_dim]);
  }
  embeddings.clear();
  state_buffer_.resize(hidden_dim);
  hidden_buffer_.resize(hidden_dim);
}

const float *NativeActorPolicy::cached_gate(DecisionMode mode,
                                            std::size_t gate_id) const {
  require(gate_id < gate_count_, "RL gate ID is out of range");
  const std::vector<float> &cache =
      mode == DecisionMode::BACKTRACE ? backtrace_cache_ : propagation_cache_;
  return &cache[gate_id * actor_.hidden_dimension()];
}

int NativeActorPolicy::select(const DecisionRequest &request) {
  require(request.candidate_ids != nullptr && request.candidate_count > 1,
          "Policy should only be called for multi-candidate decisions");
  if (actor_.is_v2()) {
    require(request.mode == DecisionMode::BACKTRACE,
            "V2 actor supports backtrace_rl only");
    require(request.candidate_count == 2,
            "V2 actor requires a binary gate with two available inputs");
    require(request.objective_id < gate_count_,
            "V2 objective gate ID is out of range");
    require(request.objective_value == 0 || request.objective_value == 1,
            "V2 objective value must be 0 or 1");
    const std::size_t cache_key =
        request.objective_id * 2 +
        static_cast<std::size_t>(request.objective_value);
    const std::size_t offset = request.objective_id * 4 +
                               static_cast<std::size_t>(request.objective_value) * 2;
    if (!v2_cache_valid_[cache_key]) {
      actor_.backtrace_action_logits_into(
          &v2_embedding_cache_[request.objective_id *
                               actor_.embedding_dimension()],
          request.objective_value, state_buffer_.data(), hidden_buffer_.data(),
          &v2_logits_cache_[offset]);
      v2_cache_valid_[cache_key] = 1;
    }
    return v2_logits_cache_[offset + 1] > v2_logits_cache_[offset] ? 1 : 0;
  }

  const std::size_t hidden_dim = actor_.hidden_dimension();
  const float *state = nullptr;
  if (request.mode == DecisionMode::BACKTRACE) {
    state = cached_gate(DecisionMode::BACKTRACE, request.objective_id);
  } else {
    std::fill(state_buffer_.begin(), state_buffer_.end(), 0.0f);
    for (std::size_t index = 0; index < request.candidate_count; ++index) {
      const std::size_t gate_id = request.candidate_ids[index];
      const float *candidate = cached_gate(DecisionMode::PROPAGATION, gate_id);
      for (std::size_t col = 0; col < hidden_dim; ++col) {
        state_buffer_[col] += candidate[col];
      }
    }
    const float divisor = static_cast<float>(request.candidate_count);
    for (float &value : state_buffer_) {
      value /= divisor;
    }
    state = state_buffer_.data();
  }

  std::size_t best = 0;
  float best_score = actor_.score_encoded(
      request.mode, state, cached_gate(request.mode, request.candidate_ids[0]),
      hidden_buffer_);
  for (std::size_t i = 1; i < request.candidate_count; ++i) {
    const float score = actor_.score_encoded(
        request.mode, state,
        cached_gate(request.mode, request.candidate_ids[i]), hidden_buffer_);
    if (score > best_score) {
      best = i;
      best_score = score;
    }
  }
  return static_cast<int>(best);
}

std::string fnv1a_file_hash(const std::string &path) {
  std::ifstream input = open_artifact(path, std::ios::in | std::ios::binary);
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

RlMode parse_rl_mode(const std::string &value) {
  if (value == "backtrace_rl") {
    return RlMode::BACKTRACE_RL;
  }
  if (value == "propagate_rl") {
    return RlMode::PROPAGATE_RL;
  }
  if (value == "both_rl") {
    return RlMode::BOTH_RL;
  }
  throw std::runtime_error(
      "Unknown RL mode '" + value +
      "'; expected backtrace_rl, propagate_rl, or both_rl");
}

std::string rl_mode_name(RlMode mode) {
  if (mode == RlMode::BACKTRACE_RL) {
    return "backtrace_rl";
  }
  if (mode == RlMode::PROPAGATE_RL) {
    return "propagate_rl";
  }
  return "both_rl";
}

bool rl_mode_enables(RlMode mode, DecisionMode decision) {
  if (mode == RlMode::BOTH_RL) {
    return true;
  }
  if (decision == DecisionMode::BACKTRACE) {
    return mode == RlMode::BACKTRACE_RL;
  }
  return mode == RlMode::PROPAGATE_RL;
}

} // namespace smartatpg
