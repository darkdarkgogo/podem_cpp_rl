#include "rl_policy.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: test_rl_policy ACTOR EMBEDDINGS CIRCUIT\n";
    return 2;
  }

  smartatpg::ActorModel actor;
  actor.load(argv[1]);
  const std::vector<float> objective = {1.0f, 0.0f};
  const std::vector<std::vector<float> > candidates = {
      {0.0f, 1.0f}, {1.0f, 1.0f}};
  const std::vector<float> backtrace =
      actor.backtrace_logits(objective, candidates);
  const std::vector<float> propagation = actor.propagation_logits(candidates);
  assert(backtrace.size() == 2 && propagation.size() == 2);
  assert(std::fabs(backtrace[0]) < 1e-7f && std::fabs(backtrace[1]) < 1e-7f);
  assert(std::fabs(propagation[0]) < 1e-7f &&
         std::fabs(propagation[1]) < 1e-7f);

  const std::string hash = smartatpg::fnv1a_file_hash(argv[3]);
  smartatpg::EmbeddingTable embeddings;
  embeddings.load(argv[2], hash);
  assert(embeddings.dimension() == 2);
  assert(embeddings.size() == 6);
  assert(embeddings.at("n1").size() == 2);

  smartatpg::NativeActorPolicy policy(argv[2], argv[1], hash);
  smartatpg::DecisionRequest request;
  request.mode = smartatpg::DecisionMode::BACKTRACE;
  request.objective_name = "n1";
  request.objective_value = 1;
  request.candidate_names = {"a", "b"};
  request.sequence = 1;
  request.fault_id = "test";
  request.backtracks = 0;
  assert(policy.select(request) == 0);
  return 0;
}
