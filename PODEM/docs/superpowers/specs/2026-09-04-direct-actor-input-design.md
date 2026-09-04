# Direct Actor Input

This amendment supersedes earlier descriptions of a shared pre-encoder and
objective-value embedding for the new comparison models. The user approved
direct graph embeddings for SmartATPG and scalar concatenation for agentATPG.

- Both graph encoders still output 11 dimensions per node.
- SmartATPG feeds this vector directly to Actor and Critic (input width 11).
- agentATPG appends the current scalar objective value, 0 or 1, and feeds the
  resulting 12 values directly to Actor and Critic.
- Neither model contains a gate_encoder or objective_value_embedding module.
- Actor and Critic retain their own existing 32-unit hidden layer. There is no
  separate projection between the graph output and these networks.
- Action masks remain separate and are applied after Actor logits. Route-lock
  and backtracking semantics are unchanged by this amendment.
- New exports use model V7, embedding V5, and benchmark bundle V4. The bundle
  records per-model Actor input widths instead of a shared width.
- Earlier V5/V6 inference models remain readable. Old pre-encoder training
  checkpoints cannot resume as the new architecture and are explicitly rejected.
- Tests must capture the actual Actor/Critic input, verify objective-value
  concatenation only for agentATPG, and check native/portable parity and PPO.

These are project-specific comparison configurations, not a claim of exact
reproduction of the original SmartATPG paper.
