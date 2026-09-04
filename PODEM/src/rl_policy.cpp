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

std::vector<std::string> required_tensor_names(
    int version, const std::string &encoder_variant) {
  std::vector<std::string> names = {
      "gate_encoder.0.weight",
      "gate_encoder.0.bias",
      "objective_value_embedding.weight",
      "backtrace_actor.0.weight",
      "backtrace_actor.0.bias",
      "backtrace_actor.2.weight",
      "backtrace_actor.2.bias",
  };
  if (version >= 7) names.erase(names.begin(), names.begin() + 3);
  if (version == 5 || encoder_variant == "fanin_mean") {
    names.insert(names.begin(), "graph_encoder.layer.bias");
    names.insert(names.begin(), "graph_encoder.layer.weight");
  } else if (encoder_variant == "level_gat_gru") {
    const char *directions[] = {"forward_pass", "reverse_pass"};
    const char *suffixes[] = {"attention", "projection.weight",
                              "gru.weight_ih", "gru.weight_hh",
                              "gru.bias_ih", "gru.bias_hh"};
    for (const char *direction : directions) {
      for (const char *suffix : suffixes) {
        names.push_back(std::string("graph_encoder.") + direction + "." + suffix);
      }
    }
  }
  return names;
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

void read_smartatpg_11d_metadata(
    std::istream &input, std::string &backend, std::string &schema,
    std::string &graph_config, std::size_t &gate_embedding_dim,
    std::size_t &policy_state_dim, std::string &snapshot) {
  std::string key;
  require(static_cast<bool>(input >> key >> backend) && key == "backend" &&
              backend == "smartatpg", "Invalid artifact backend");
  require(static_cast<bool>(input >> key >> schema) &&
              key == "feature_schema" &&
              schema == "SMARTATPG_FEATURES_V2_11D",
          "Invalid SmartATPG 11D feature schema");
  require(static_cast<bool>(input >> key >> graph_config) &&
              key == "graph_config" &&
              graph_config == "fanin_mean_1x22x11",
          "Invalid SmartATPG GraphSAGE configuration");
  require(static_cast<bool>(input >> key >> gate_embedding_dim) &&
              key == "gate_embedding_dim" && gate_embedding_dim == 11,
          "SmartATPG gate embedding dimension must be 11");
  require(static_cast<bool>(input >> key >> policy_state_dim) &&
              key == "policy_state_dim" && policy_state_dim == 13,
          "SmartATPG policy state dimension must be 13");
  require(static_cast<bool>(input >> key >> snapshot) && key == "snapshot" &&
              snapshot.size() == 64 &&
              snapshot.find_first_not_of("0123456789abcdef") ==
                  std::string::npos,
          "Invalid artifact snapshot identifier");
}

void read_smartatpg_v6_metadata(
    std::istream &input, std::string &backend, std::string &schema,
    std::string &encoder_variant, std::string &graph_config,
    std::size_t &gate_embedding_dim, std::size_t &actor_input_dim,
    std::size_t &action_mask_dim, std::size_t &decision_state_dim,
    std::string &snapshot, bool direct_actor = false, bool has_co = false) {
  std::string key;
  const std::size_t expected_gate_dim = has_co ? 12U : 11U;
  const std::string expected_schema = has_co ? "SMARTATPG_FEATURES_V3_12D_CO" :
                                              "SMARTATPG_FEATURES_V2_11D";
  require(static_cast<bool>(input >> key >> backend) && key == "backend" &&
              backend == "smartatpg", "Invalid artifact backend");
  require(static_cast<bool>(input >> key >> schema) &&
              key == "feature_schema" &&
              schema == expected_schema,
          "Invalid SmartATPG feature schema for model version");
  require(static_cast<bool>(input >> key >> encoder_variant) &&
              key == "encoder_variant" &&
              (encoder_variant == "fanin_mean" ||
               encoder_variant == "level_gat_gru"),
          "Invalid SmartATPG encoder variant");
  require(static_cast<bool>(input >> key >> graph_config) &&
              key == "graph_config",
          "Missing SmartATPG graph configuration");
  const std::string expected_config =
      encoder_variant == "fanin_mean" ?
          (has_co ? "fanin_mean_1x24x12" : "fanin_mean_1x22x11") :
          (has_co ? "level_gat_gru_fwd_rev_12d_v2" : "level_gat_gru_fwd_rev_11d_v1");
  require(graph_config == expected_config,
          "SmartATPG encoder variant and graph configuration do not match");
  require(static_cast<bool>(input >> key >> gate_embedding_dim) &&
              key == "gate_embedding_dim" && gate_embedding_dim == expected_gate_dim,
          "SmartATPG gate embedding dimension does not match model version");
  require(static_cast<bool>(input >> key >> actor_input_dim) &&
              key == "actor_input_dim" &&
              actor_input_dim == expected_gate_dim +
                  (direct_actor && encoder_variant == "level_gat_gru" ? 1U : 0U),
          "Actor input dimension does not match its architecture");
  require(static_cast<bool>(input >> key >> action_mask_dim) &&
              key == "action_mask_dim" && action_mask_dim == 2,
          "SmartATPG action mask dimension must be 2");
  require(static_cast<bool>(input >> key >> decision_state_dim) &&
              key == "decision_state_dim" && decision_state_dim == actor_input_dim + 2,
          "Decision state dimension must match Actor input plus mask");
  require(static_cast<bool>(input >> key >> snapshot) && key == "snapshot" &&
              snapshot.size() == 64 &&
              snapshot.find_first_not_of("0123456789abcdef") ==
                  std::string::npos,
          "Invalid artifact snapshot identifier");
}

} // namespace

void EmbeddingTable::load(const std::string &path,
                          const std::string &expected_circuit_hash) {
  std::ifstream input = open_artifact(path);
  require(input.good(), "Cannot open embedding file: " + path);

  std::string header;
  std::getline(input, header);
  require(header == "SMARTATPG_EMBEDDINGS_V2" ||
              header == "SMARTATPG_EMBEDDINGS_V3" ||
              header == "SMARTATPG_EMBEDDINGS_V4" || header == "SMARTATPG_EMBEDDINGS_V5" ||
              header == "SMARTATPG_EMBEDDINGS_V6",
          "Unsupported embedding format in: " + path);
  backend_ = "smartatpg";
  schema_.clear();
  graph_config_.clear();
  encoder_variant_.clear();
  snapshot_.clear();
  policy_state_dim_ = 0;
  actor_input_dim_ = 0;
  action_mask_dim_ = 0;
  decision_state_dim_ = 0;
  if (header == "SMARTATPG_EMBEDDINGS_V2") {
    require(false,
            "Legacy 80-dimensional SmartATPG descriptors are incompatible "
            "with the 11-dimensional GraphSAGE format");
  } else if (header == "SMARTATPG_EMBEDDINGS_V3") {
    std::size_t declared_gate_dim = 0;
    read_smartatpg_11d_metadata(
        input, backend_, schema_, graph_config_, declared_gate_dim,
        policy_state_dim_, snapshot_);
    encoder_variant_ = "fanin_mean";
    actor_input_dim_ = policy_state_dim_;
    action_mask_dim_ = 2;
    decision_state_dim_ = policy_state_dim_;
  } else if (header == "SMARTATPG_EMBEDDINGS_V4" || header == "SMARTATPG_EMBEDDINGS_V5" ||
             header == "SMARTATPG_EMBEDDINGS_V6") {
    std::size_t declared_gate_dim = 0;
    read_smartatpg_v6_metadata(
        input, backend_, schema_, encoder_variant_, graph_config_,
        declared_gate_dim, actor_input_dim_, action_mask_dim_,
        decision_state_dim_, snapshot_, header != "SMARTATPG_EMBEDDINGS_V4",
        header == "SMARTATPG_EMBEDDINGS_V6");
    policy_state_dim_ = decision_state_dim_;
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
  require(dimension_ == (header == "SMARTATPG_EMBEDDINGS_V6" ? 12U : 11U),
          "SmartATPG gate embedding dimension does not match artifact version");
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
    throw std::runtime_error("Missing gate embedding for wire: " + name);
  }
  return found->second;
}

void EmbeddingTable::clear() {
  embeddings_.clear();
  embeddings_.rehash(0);
  dimension_ = 0;
  actor_input_dim_ = 0;
  action_mask_dim_ = 0;
  decision_state_dim_ = 0;
}

void ActorModel::load(const std::string &path) {
  std::ifstream input = open_artifact(path);
  require(input.good(), "Cannot open actor file: " + path);

  std::string header;
  std::string key;
  std::getline(input, header);
  version_ = header == "SMARTATPG_ACTOR_V3" ? 3 :
                    (header == "SMARTATPG_ACTOR_V4" ? 4 :
                     (header == "SMARTATPG_MODEL_V5" ? 5 :
                      (header == "SMARTATPG_MODEL_V6" ? 6 :
                       (header == "SMARTATPG_MODEL_V7" ? 7 :
                        (header == "SMARTATPG_MODEL_V8" ? 8 : 0)))));
  require(version_ != 0,
          "Unsupported actor format in: " + path);
  backend_ = "smartatpg";
  schema_.clear();
  graph_config_.clear();
  encoder_variant_.clear();
  snapshot_.clear();
  gate_embedding_dim_ = 0;
  action_mask_dim_ = 0;
  decision_state_dim_ = 0;
  if (version_ == 3) {
    read_backend_metadata(input, backend_, schema_, snapshot_);
    require(false,
            "Legacy 80-dimensional SmartATPG actors are incompatible with "
            "the 11-dimensional GraphSAGE format");
  } else if (version_ == 4) {
    require(false,
            "SMARTATPG_ACTOR_V4 does not contain GraphSAGE weights; export "
            "a SMARTATPG_MODEL_V5 model");
  } else if (version_ == 5) {
    read_smartatpg_11d_metadata(
        input, backend_, schema_, graph_config_, gate_embedding_dim_,
        embedding_dim_, snapshot_);
    encoder_variant_ = "fanin_mean";
    action_mask_dim_ = 2;
    decision_state_dim_ = embedding_dim_;
    int best_round = 0;
    std::string best_score;
    require(static_cast<bool>(input >> key >> best_round) &&
                key == "best_round" && best_round >= 0,
            "Invalid SmartATPG best round in: " + path);
    require(static_cast<bool>(input >> key >> best_score) &&
                key == "best_score" && !best_score.empty(),
            "Invalid SmartATPG best score in: " + path);
  } else if (version_ >= 6) {
    read_smartatpg_v6_metadata(
        input, backend_, schema_, encoder_variant_, graph_config_,
        gate_embedding_dim_, embedding_dim_, action_mask_dim_,
        decision_state_dim_, snapshot_, version_ >= 7, version_ == 8);
    int best_round = 0;
    std::string best_score;
    require(static_cast<bool>(input >> key >> best_round) &&
                key == "best_round" && best_round >= 0,
            "Invalid SmartATPG best round in: " + path);
    require(static_cast<bool>(input >> key >> best_score) &&
                key == "best_score" && !best_score.empty(),
            "Invalid SmartATPG best score in: " + path);
  }

  if (version_ < 4) {
    require(static_cast<bool>(input >> key >> embedding_dim_) &&
                key == "embedding_dim" && embedding_dim_ > 0,
            "Invalid actor embedding dimension in: " + path);
    gate_embedding_dim_ = embedding_dim_;
  }
  require(static_cast<bool>(input >> key >> hidden_dim_) && key == "hidden_dim" &&
              hidden_dim_ > 0,
          "Invalid actor hidden dimension in: " + path);
  require(version_ < 4 ||
              (version_ == 5 && embedding_dim_ == 13) ||
              (version_ == 6 && embedding_dim_ == 11) || version_ >= 7,
          "SmartATPG Actor input dimension does not match its model version");

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

  const std::vector<std::string> names =
      required_tensor_names(version_, encoder_variant_);
  require(tensors_.size() == names.size(), "Unexpected actor tensor count");
  for (const std::string &name : names) {
    tensor(name);
  }

  if (encoder_variant_ == "fanin_mean") {
    require(tensor("graph_encoder.layer.weight").rows == gate_embedding_dim_ &&
                tensor("graph_encoder.layer.weight").cols == 2 * gate_embedding_dim_,
            "GraphSAGE weight dimensions do not match gate embedding dimension");
    require(tensor("graph_encoder.layer.bias").values.size() == gate_embedding_dim_,
            "GraphSAGE bias dimensions do not match gate embedding dimension");
  }
  if (encoder_variant_ == "level_gat_gru") {
    const char *directions[] = {"forward_pass", "reverse_pass"};
    for (const char *direction : directions) {
      const std::string prefix = std::string("graph_encoder.") + direction + ".";
      require(tensor(prefix + "attention").values.size() == 2 * gate_embedding_dim_,
              "GAT attention dimensions do not match gate embedding dimension");
      require(tensor(prefix + "projection.weight").rows == gate_embedding_dim_ &&
                  tensor(prefix + "projection.weight").cols == gate_embedding_dim_,
              "GAT projection dimensions do not match gate embedding dimension");
      require(tensor(prefix + "gru.weight_ih").rows == 3 * gate_embedding_dim_ &&
                  tensor(prefix + "gru.weight_ih").cols == gate_embedding_dim_ &&
                  tensor(prefix + "gru.weight_hh").rows == 3 * gate_embedding_dim_ &&
                  tensor(prefix + "gru.weight_hh").cols == gate_embedding_dim_,
              "GAT-GRU weight dimensions do not match gate embedding dimension");
      require(tensor(prefix + "gru.bias_ih").values.size() == 3 * gate_embedding_dim_ &&
                  tensor(prefix + "gru.bias_hh").values.size() == 3 * gate_embedding_dim_,
              "GAT-GRU bias dimensions do not match gate embedding dimension");
    }
  }

  gate_weight_ = nullptr;
  gate_bias_ = nullptr;
  objective_value_embedding_ = nullptr;
  if (version_ < 7) {
  require(tensor("gate_encoder.0.weight").rows == hidden_dim_ &&
              tensor("gate_encoder.0.weight").cols == embedding_dim_,
          "gate_encoder weight dimensions do not match actor header");
  require(tensor("gate_encoder.0.bias").rows *
                  tensor("gate_encoder.0.bias").cols ==
              hidden_dim_,
          "gate_encoder bias dimensions do not match actor header");
  gate_weight_ = &tensor("gate_encoder.0.weight");
  gate_bias_ = &tensor("gate_encoder.0.bias");
  const Tensor &value_embedding = tensor("objective_value_embedding.weight");
  require(value_embedding.rows == 2 && value_embedding.cols == hidden_dim_,
          "objective_value_embedding dimensions must be [2, hidden_dim]");
  objective_value_embedding_ = &value_embedding;
  }
  require(tensor("backtrace_actor.0.weight").rows == hidden_dim_ &&
              tensor("backtrace_actor.0.weight").cols == (version_ >= 7 ? embedding_dim_ : hidden_dim_),
          "V2 backtrace hidden weight dimensions are invalid");
  require(tensor("backtrace_actor.0.bias").values.size() == hidden_dim_,
          "V2 backtrace hidden bias dimensions are invalid");
  require(tensor("backtrace_actor.2.weight").rows == 2 &&
              tensor("backtrace_actor.2.weight").cols == hidden_dim_,
          "V2 backtrace output weight dimensions must be [2, hidden_dim]");
  require(tensor("backtrace_actor.2.bias").values.size() == 2,
          "V2 backtrace output bias dimensions must be [2]");
}

const ActorModel::Tensor &ActorModel::tensor(const std::string &name) const {
  const auto found = tensors_.find(name);
  if (found == tensors_.end()) {
    throw std::runtime_error("Missing actor tensor: " + name);
  }
  return found->second;
}

std::vector<float> ActorModel::backtrace_action_logits(
    const std::vector<float> &objective, int objective_value) const {
  std::vector<float> state(hidden_dim_);
  std::vector<float> hidden(hidden_dim_);
  std::vector<float> logits(2);
  require(objective.size() == (version_ >= 7 ? gate_embedding_dim_ : embedding_dim_),
          "Policy state dimension does not match V2 actor");
  std::vector<float> input = objective;
  if (version_ >= 7 && encoder_variant_ == "level_gat_gru")
    input.push_back(static_cast<float>(objective_value));
  backtrace_action_logits_into(input.data(), objective_value, state.data(),
                               hidden.data(), logits.data());
  return logits;
}

void ActorModel::backtrace_action_logits_into(
    const float *objective, int objective_value, float *state, float *hidden,
    float *logits) const {
  require(objective_value == 0 || objective_value == 1,
          "Backtrace objective value must be 0 or 1");
  require(objective != nullptr && state != nullptr && hidden != nullptr &&
              logits != nullptr,
          "V2 actor buffers must not be null");
  if (version_ < 7) {
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

  }
  const float *actor_input = version_ >= 7 ? objective : state;
  const std::size_t actor_width = version_ >= 7 ? embedding_dim_ : hidden_dim_;
  const Tensor &hidden_weight = tensor("backtrace_actor.0.weight");
  const Tensor &hidden_bias = tensor("backtrace_actor.0.bias");
  for (std::size_t row = 0; row < hidden_dim_; ++row) {
    float value = hidden_bias.values[row];
    const std::size_t offset = row * actor_width;
    for (std::size_t col = 0; col < actor_width; ++col) {
      value += hidden_weight.values[offset + col] * actor_input[col];
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
  require(expected_backend.empty() || expected_backend == "smartatpg",
          "Unknown embedding backend");
  require(expected_backend.empty() || expected_backend == actor_.backend(),
          "Requested embedding backend conflicts with actor backend");
  require(embeddings.backend() == actor_.backend(), "Embedding and actor backend mismatch");
  require(embeddings.schema() == actor_.schema(), "Embedding and actor schema mismatch");
  require(embeddings.graph_config() == actor_.graph_config(),
          "Embedding and actor graph configuration mismatch");
  require(embeddings.encoder_variant() == actor_.encoder_variant(),
          "Embedding and actor encoder variant mismatch");
  require(embeddings.snapshot() == actor_.snapshot(), "Embedding and actor snapshot mismatch");
  require(embeddings.dimension() == actor_.gate_embedding_dimension(),
          "SmartATPG gate embedding dimensions do not match");
  require(embeddings.actor_input_dimension() == actor_.embedding_dimension(),
          "SmartATPG Actor input dimensions do not match");
  require(embeddings.action_mask_dimension() == actor_.action_mask_dimension(),
          "SmartATPG action mask dimensions do not match");
  require(embeddings.decision_state_dimension() ==
              actor_.decision_state_dimension(),
          "SmartATPG decision state dimensions do not match");
  require(!gate_names_by_id.empty(), "Circuit gate table must not be empty");
  require(std::unordered_set<std::string>(gate_names_by_id.begin(), gate_names_by_id.end()).size() == gate_names_by_id.size(),
          "Circuit gate table contains duplicate wire names");
  require(embeddings.size() == gate_names_by_id.size(),
          "SmartATPG descriptor count does not match circuit wires");

  gate_count_ = gate_names_by_id.size();
  const std::size_t gate_embedding_dim = actor_.gate_embedding_dimension();
  v2_mask_is_actor_input_ =
      actor_.version_ == 5;
  v2_variants_per_gate_ = v2_mask_is_actor_input_ ? 8 : 2;
  v2_embedding_cache_.resize(gate_count_ * gate_embedding_dim);
  v2_logits_cache_.resize(gate_count_ * v2_variants_per_gate_ * 2);
  v2_cache_valid_.assign(gate_count_ * v2_variants_per_gate_, 0);
  for (std::size_t gate_id = 0; gate_id < gate_count_; ++gate_id) {
    const std::vector<float> &embedding =
        embeddings.at(gate_names_by_id[gate_id]);
    std::copy(embedding.begin(), embedding.end(),
              v2_embedding_cache_.begin() + gate_id * gate_embedding_dim);
  }
  embeddings.clear();
  v2_policy_input_buffer_.resize(actor_.embedding_dimension());
  state_buffer_.resize(actor_.hidden_dimension());
  hidden_buffer_.resize(actor_.hidden_dimension());
}

int NativeActorPolicy::select(const DecisionRequest &request) {
  require(request.candidate_ids != nullptr && request.candidate_count > 1,
          "Policy should only be called for multi-candidate decisions");
  require(request.mode == DecisionMode::BACKTRACE,
          "SmartATPG actor supports backtrace_rl only");
  require(request.candidate_count == 2,
          "SmartATPG actor requires a binary gate with two available inputs");
  require(request.objective_id < gate_count_,
          "SmartATPG objective gate ID is out of range");
  require(request.objective_value == 0 || request.objective_value == 1,
          "SmartATPG objective value must be 0 or 1");
  const std::size_t mask_code =
      (request.action_mask[0] ? 1U : 0U) |
      (request.action_mask[1] ? 2U : 0U);
  require(mask_code != 0, "SmartATPG action mask must enable at least one input");
  const std::size_t variant = v2_mask_is_actor_input_
      ? static_cast<std::size_t>(request.objective_value) * 4 + mask_code
      : static_cast<std::size_t>(request.objective_value);
  const std::size_t cache_key =
      request.objective_id * v2_variants_per_gate_ + variant;
  const std::size_t offset = cache_key * 2;
  if (!v2_cache_valid_[cache_key]) {
    const std::size_t gate_embedding_dim = actor_.gate_embedding_dimension();
    const float *embedding =
        &v2_embedding_cache_[request.objective_id * gate_embedding_dim];
    std::copy(embedding, embedding + gate_embedding_dim,
              v2_policy_input_buffer_.begin());
    if (v2_mask_is_actor_input_) {
      v2_policy_input_buffer_[gate_embedding_dim] =
          request.action_mask[0] ? 1.0f : 0.0f;
      v2_policy_input_buffer_[gate_embedding_dim + 1] =
          request.action_mask[1] ? 1.0f : 0.0f;
    } else if (actor_.version_ >= 7 && actor_.encoder_variant() == "level_gat_gru") {
      v2_policy_input_buffer_[gate_embedding_dim] = static_cast<float>(request.objective_value);
    }
    actor_.backtrace_action_logits_into(
        v2_policy_input_buffer_.data(), request.objective_value,
        state_buffer_.data(), hidden_buffer_.data(), &v2_logits_cache_[offset]);
    v2_cache_valid_[cache_key] = 1;
  }
  if (!request.action_mask[0]) {
    return 1;
  }
  if (!request.action_mask[1]) {
    return 0;
  }
  return v2_logits_cache_[offset + 1] > v2_logits_cache_[offset] ? 1 : 0;
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
