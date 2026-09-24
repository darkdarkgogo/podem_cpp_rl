"""Python utilities for the C++ RL-guided PODEM engine.

Heavy PyTorch modules are imported lazily so data preparation and portable
inference remain usable in lightweight environments.
"""

from importlib import import_module


_EXPORTS = {
    "CppPodemBacktraceV2Trainer": (".cpp_bridge", "CppPodemBacktraceV2Trainer"),
    "CppPodemBacktraceV2Evaluator": (".cpp_bridge", "CppPodemBacktraceV2Evaluator"),
    "catalog_cpp_podem": (".cpp_bridge", "catalog_cpp_podem"),
    "export_actor_v2_state_dict": (".cpp_bridge", "export_actor_v2_state_dict"),
    "profile_cpp_podem": (".cpp_bridge", "profile_cpp_podem"),
    "GAT_REWARD_SCHEME": (".smartatpg_rewards", "GAT_REWARD_SCHEME"),
    "MEAN_REWARD_SCHEME": (".smartatpg_rewards", "MEAN_REWARD_SCHEME"),
    "reward_scheme_for_encoder": (".smartatpg_rewards", "reward_scheme_for_encoder"),
    "smartatpg_backtrack_reward": (".smartatpg_rewards", "smartatpg_backtrack_reward"),
    "smartatpg_pi_reward": (".smartatpg_rewards", "smartatpg_pi_reward"),
    "GATGRUSmartATPGPPOAgent": (".gat_gru", "GATGRUSmartATPGPPOAgent"),
    "GATGRUSmartATPGPolicy": (".gat_gru", "GATGRUSmartATPGPolicy"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
