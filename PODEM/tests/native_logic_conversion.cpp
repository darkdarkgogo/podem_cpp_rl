// Reuse CLI helper definitions without its entry point.
#define main atpg_cli_main
#include "../src/main.cpp"
#undef main

// Explicit instantiation permits testing private methods without changing
// ATPG's access specifiers (which affect MSVC symbol names).
template <typename Tag, typename Tag::Member member>
struct MethodAccess {
  friend typename Tag::Member access(Tag) { return member; }
};

struct Itoc {
  using Member = char (ATPG::*)(const int &);
  friend Member access(Itoc);
};
struct Ctoi {
  using Member = int (ATPG::*)(const char &);
  friend Member access(Ctoi);
};
struct FindType {
  using Member = int (ATPG::*)(const std::string &);
  friend Member access(FindType);
};
template struct MethodAccess<Itoc, &ATPG::itoc>;
template struct MethodAccess<Ctoi, &ATPG::ctoi>;
template struct MethodAccess<FindType, &ATPG::FindType>;

class ATPGScoapTestAccess {
 public:
  static void load(ATPG &atpg, const std::string &path) {
    atpg.set_SCOAP(true);
    atpg.input(path);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
  }

  static std::string select_control(
      ATPG &atpg, const std::string &gate_name, int target, bool easiest) {
    ATPG::wptr output = atpg.wfind(gate_name);
    if (!output || output->inode.empty()) return "missing";
    ATPG::nptr gate = output->inode.front();
    for (ATPG::wptr wire : gate->iwire) wire->value = U;
    ATPG::wptr selected = easiest
        ? atpg.find_easiest_control(gate, target)
        : atpg.find_hardest_control(gate, target);
    return selected ? selected->name : "none";
  }

  static std::string select_propagation(ATPG &atpg) {
    ATPG::wptr stem = atpg.wfind("a");
    ATPG::wptr first = atpg.wfind("p");
    ATPG::wptr second = atpg.wfind("q");
    if (!stem || !first || !second) return "missing";
    for (ATPG::wptr wire : atpg.sort_wlist) wire->value = U;
    stem->value = D;
    first->inode.front()->set_marked();
    second->inode.front()->set_marked();
    ATPG::nptr selected = atpg.find_propagate_gate(stem->level);
    if (!selected) return "none";
    const int first_co = atpg.co[first->wlist_index];
    const int second_co = atpg.co[second->wlist_index];
    return selected->owire.front()->name + ":" + std::to_string(first_co)
        + ":" + std::to_string(second_co);
  }
};

int main(int argc, char **argv) {
  ATPG atpg;
  if (argc != 3) return 2;
  const std::string operation = argv[1];
  if (operation == "itoc") {
    std::cout << (atpg.*access(Itoc{}))(std::stoi(argv[2]));
  } else if (operation == "ctoi") {
    std::cout << (atpg.*access(Ctoi{}))(argv[2][0]);
  } else if (operation == "gate") {
    std::cout << (atpg.*access(FindType{}))(argv[2]);
  } else if (operation == "undetected-output") {
    atpg.detected_num = 1;
    atpg.set_backtrack_limit(0);
    atpg.input(argv[2]);
    atpg.level_circuit();
    atpg.rearrange_gate_inputs();
    atpg.create_dummy_gate();
    atpg.generate_fault_list();
    atpg.test();
  } else if (operation.rfind("scoap-control-", 0) == 0) {
    ATPGScoapTestAccess::load(atpg, argv[2]);
    const bool easiest = operation.find("easy") != std::string::npos;
    const int target = operation.back() - '0';
    std::cout << ATPGScoapTestAccess::select_control(
        atpg, "y", target, easiest);
  } else if (operation == "scoap-propagation") {
    ATPGScoapTestAccess::load(atpg, argv[2]);
    std::cout << ATPGScoapTestAccess::select_propagation(atpg);
  } else {
    return 2;
  }
  return 0;
}
