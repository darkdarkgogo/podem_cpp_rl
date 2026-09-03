"""Build the standalone executable with the locally configured C++ toolchain."""

import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("build/atpg_rl_smartatpg")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [
        str(path) for path in sorted(Path("src").glob("*.cpp"))
        if path.name != "python_bindings.cpp"
    ]
    if os.name != "nt":
        compiler = os.environ.get("CXX", "g++")
        command = [
            compiler, "-std=c++11", "-O2", "-Isrc", *sources,
            "-o", str(args.output),
        ]
        subprocess.run(command, check=True)
        print(f"BUILT {args.output.resolve()}")
        return

    from setuptools._distutils.ccompiler import new_compiler
    from setuptools._distutils.sysconfig import customize_compiler

    compiler = new_compiler()
    customize_compiler(compiler)
    flags = ["/std:c++14", "/EHsc", "/O2"]
    objects = compiler.compile(sources, output_dir="build/smartatpg_obj", include_dirs=["src"],
                               extra_postargs=flags)
    link_output = args.output
    if link_output.suffix.lower() == ".exe":
        link_output = link_output.with_suffix("")
    link_args = ["/MANIFEST:NO"]
    compiler.link_executable(
        objects, str(link_output), extra_postargs=link_args
    )
    actual_output = (
        link_output.with_suffix(".exe")
        if compiler.compiler_type == "msvc" else link_output
    )
    print(f"BUILT {actual_output.resolve()}")


if __name__ == "__main__":
    main()
