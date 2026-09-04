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
  } else {
    return 2;
  }
  return 0;
}
