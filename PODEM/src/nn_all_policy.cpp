#include "nn_all_policy.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <unordered_map>

#if defined(_WIN32) && !defined(__MINGW32__)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace smartatpg {
namespace {

std::ifstream open_text(const std::string &path) {
#if defined(_WIN32) && !defined(__MINGW32__)
  const UINT codepage = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
      path.c_str(), -1, nullptr, 0) > 0 ? CP_UTF8 : CP_ACP;
  const int count = MultiByteToWideChar(codepage, 0, path.c_str(), -1, nullptr, 0);
  if (count <= 0) throw std::runtime_error("Invalid NN-All artifact path");
  std::vector<wchar_t> wide(static_cast<std::size_t>(count));
  if (!MultiByteToWideChar(codepage, 0, path.c_str(), -1, wide.data(), count))
    throw std::runtime_error("Cannot decode NN-All artifact path");
  return std::ifstream(wide.data());
#else
  return std::ifstream(path.c_str());
#endif
}

void expect_key(std::istream &input, const char *expected) {
  std::string actual;
  if (!(input >> actual) || actual != expected)
    throw std::runtime_error(std::string("Invalid NN-All artifact field: ") + expected);
}

template <std::size_t N>
void read_array(std::istream &input, const char *name, std::array<float, N> &values) {
  expect_key(input, name);
  std::size_t count = 0;
  if (!(input >> count) || count != N)
    throw std::runtime_error(std::string("Invalid NN-All tensor size: ") + name);
  for (float &value : values) {
    if (!(input >> value) || !std::isfinite(value))
      throw std::runtime_error(std::string("Invalid NN-All tensor value: ") + name);
  }
}

float sigmoid(float input) {
  if (input >= 0) {
    const float factor = std::exp(-input);
    return 1.0f / (1.0f + factor);
  }
  const float factor = std::exp(input);
  return factor / (1.0f + factor);
}

const char *const types[] = {"AND", "NAND", "OR", "NOR", "PI", "BUF", "NOT"};

int type_index(const std::string &name) {
  for (int index = 0; index < 7; ++index)
    if (name == types[index]) return index;
  throw std::runtime_error("Unknown NN-All gate type: " + name);
}

} // namespace

NnAllModel::NnAllModel(const std::string &path) {
  std::ifstream input = open_text(path);
  if (!input.good()) throw std::runtime_error("Cannot open NN-All model: " + path);
  expect_key(input, "NN_ALL_PODEM_MODEL_V2");
  expect_key(input, "feature_schema");
  std::string schema;
  if (!(input >> schema) || schema != "NN_ALL_PODEM_PAPER_12D_V2")
    throw std::runtime_error("NN-All feature schema mismatch");
  expect_key(input, "hidden_dim");
  int hidden_dim = 0;
  if (!(input >> hidden_dim) || hidden_dim != 60)
    throw std::runtime_error("NN-All hidden dimension mismatch");
  expect_key(input, "manifest_sha256");
  if (!(input >> manifest_hash_) || manifest_hash_.size() != 64)
    throw std::runtime_error("NN-All manifest hash is invalid");
  expect_key(input, "seed");
  int seed = 0;
  if (!(input >> seed) || seed < 0)
    throw std::runtime_error("NN-All seed is invalid");
  read_array(input, "w1", w1_);
  read_array(input, "b1", b1_);
  read_array(input, "w2", w2_);
  std::array<float, 1> final_bias{};
  read_array(input, "b2", final_bias);
  b2_ = final_bias[0];
  std::string trailing;
  if (input >> trailing) throw std::runtime_error("Trailing NN-All model data");
}

float NnAllModel::score(const std::array<float, 12> &features) const {
  for (std::size_t index = 0; index < 12; ++index) {
    float value = features[index];
    if (!std::isfinite(value) || value < 0 || (index != 1 && value > 1))
      throw std::runtime_error("Invalid NN-All candidate feature");
  }
  float output = b2_;
  for (std::size_t hidden = 0; hidden < 60; ++hidden) {
    float value = b1_[hidden];
    for (std::size_t input = 0; input < 12; ++input)
      value += features[input] * w1_[hidden * 12 + input];
    output += sigmoid(value) * w2_[hidden];
  }
  return sigmoid(output);
}

NnAllPolicy::NnAllPolicy(
    const std::string &model_path, const std::string &features_path,
    const std::string &expected_circuit_hash,
    const std::vector<std::string> &gate_names_by_id)
    : model_(model_path), gates_(gate_names_by_id.size()) {
  std::ifstream input = open_text(features_path);
  if (!input.good())
    throw std::runtime_error("Cannot open NN-All features: " + features_path);
  expect_key(input, "NN_ALL_PODEM_FEATURES_V2");
  expect_key(input, "circuit_hash");
  std::string hash;
  if (!(input >> hash) || hash != expected_circuit_hash)
    throw std::runtime_error("NN-All feature circuit hash mismatch");
  expect_key(input, "feature_hash");
  std::string expected_feature_hash;
  if (!(input >> expected_feature_hash) || expected_feature_hash.size() != 16)
    throw std::runtime_error("NN-All feature hash is invalid");
  expect_key(input, "gate_count");
  std::size_t count = 0;
  if (!(input >> count) || count != gates_.size())
    throw std::runtime_error("NN-All feature gate count mismatch");
  std::unordered_map<std::string, std::size_t> indices;
  for (std::size_t index = 0; index < gate_names_by_id.size(); ++index)
    if (!indices.emplace(gate_names_by_id[index], index).second)
      throw std::runtime_error("Duplicate NN-All gate name");
  std::vector<bool> seen(count, false);
  std::string line;
  std::getline(input, line);
  std::uint64_t feature_hash = UINT64_C(14695981039346656037);
  for (std::size_t row = 0; row < count; ++row) {
    if (!std::getline(input, line))
      throw std::runtime_error("Missing NN-All feature row");
    if (!line.empty() && line.back() == '\r') line.pop_back();
    for (unsigned char byte : line) {
      feature_hash ^= byte;
      feature_hash *= UINT64_C(1099511628211);
    }
    feature_hash ^= static_cast<unsigned char>('\n');
    feature_hash *= UINT64_C(1099511628211);
    std::istringstream row_input(line);
    expect_key(row_input, "gate");
    std::string name, kind;
    Gate gate;
    if (!(row_input >> name >> kind >> gate.ldpi >> gate.sdpi >> gate.cc0 >> gate.cc1 >> gate.cop))
      throw std::runtime_error("Malformed NN-All feature row");
    std::string extra;
    if (row_input >> extra)
      throw std::runtime_error("Trailing NN-All feature row data");
    const auto found = indices.find(name);
    if (found == indices.end() || seen[found->second])
      throw std::runtime_error("NN-All feature gate name mismatch");
    gate.type = type_index(kind);
    if (!std::isfinite(gate.ldpi) || !std::isfinite(gate.sdpi) || !std::isfinite(gate.cc0) ||
        !std::isfinite(gate.cc1) || !std::isfinite(gate.cop) ||
        gate.ldpi < 0 || gate.ldpi > 1 || gate.sdpi < 0 ||
        gate.cc0 < 0 || gate.cc0 > 1 || gate.cc1 < 0 || gate.cc1 > 1 ||
        gate.cop < 0 || gate.cop > 1)
      throw std::runtime_error("Invalid NN-All gate feature");
    gates_[found->second] = gate;
    seen[found->second] = true;
  }
  std::ostringstream hash_text;
  hash_text << std::hex << std::setfill('0') << std::setw(16) << feature_hash;
  if (hash_text.str() != expected_feature_hash)
    throw std::runtime_error("NN-All feature row hash mismatch");
  std::string trailing;
  if (input >> trailing) throw std::runtime_error("Trailing NN-All feature data");
}

int NnAllPolicy::select(const DecisionRequest &request) {
  if (request.mode != DecisionMode::BACKTRACE || request.objective_id >= gates_.size() ||
      request.candidate_count > 2 || request.candidate_ids == nullptr)
    throw std::runtime_error("Invalid NN-All backtrace request");
  if (request.candidate_count == 2 &&
      request.action_mask[0] != request.action_mask[1])
    return request.action_mask[0] ? 0 : 1;
  const Gate &parent = gates_[request.objective_id];
  int selected = -1;
  float best = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0; index < request.candidate_count; ++index) {
    if (!request.action_mask[index]) continue;
    const std::size_t candidate_id = request.candidate_ids[index];
    if (candidate_id >= gates_.size())
      throw std::runtime_error("Unknown NN-All candidate gate ID");
    const std::size_t key = request.objective_id * gates_.size() + candidate_id;
    auto hit = score_cache_.find(key);
    float score = 0;
    if (hit == score_cache_.end()) {
      const Gate &candidate = gates_[candidate_id];
      std::array<float, 12> vector{{candidate.ldpi, candidate.sdpi,
          candidate.cc0, candidate.cc1, candidate.cop, 0, 0, 0, 0, 0, 0, 0}};
      vector[5 + parent.type] = 1.0f;
      score = model_.score(vector);
      score_cache_.emplace(key, score);
    } else {
      score = hit->second;
    }
    if (score > best) {
      best = score;
      selected = static_cast<int>(index);
    }
  }
  if (selected < 0) throw std::runtime_error("NN-All has no legal candidate");
  return selected;
}

} // namespace smartatpg
