"""Build the standalone executable with the locally configured C++ toolchain."""

import argparse
import os
from pathlib import Path

from setuptools._distutils.ccompiler import new_compiler
from setuptools._distutils.sysconfig import customize_compiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/atpg_rl_smartatpg.exe" if os.name == "nt" else "build/atpg_rl_smartatpg"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compiler = new_compiler()
    customize_compiler(compiler)
    flags = ["/std:c++14", "/EHsc", "/O2"] if compiler.compiler_type == "msvc" else ["-std=c++11", "-O2"]
    sources = [str(path) for path in sorted(Path("src").glob("*.cpp")) if path.name != "python_bindings.cpp"]
    objects = compiler.compile(sources, output_dir="build/smartatpg_obj", include_dirs=["src"],
                               extra_postargs=flags)
    compiler.link_executable(objects, str(args.output))
    print(f"BUILT {args.output.resolve()}")


if __name__ == "__main__":
    main()
