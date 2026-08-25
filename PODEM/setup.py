import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

CORE_SOURCES = [
    "atpg.cpp",
    "runtime_config.cpp",
    "input.cpp",
    "level.cpp",
    "sim.cpp",
    "podem.cpp",
    "rl_policy.cpp",
    "rl_atpg.cpp",
    "init_flist.cpp",
    "faultsim.cpp",
    "tdfsim.cpp",
    "tdfatpg.cpp",
    "display.cpp",
    "python_bindings.cpp",
]

DEEPGATE_PACKAGE_ROOT = "vendor/deepgate_recgnn_extractor/deepgate_recgnn_extractor"


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            compile_args = ["/std:c++14", "/EHsc", "/O2"]
        else:
            compile_args = ["-std=c++11", "-Ofast"]
        for extension in self.extensions:
            extension.extra_compile_args = compile_args
        super().build_extensions()


extension = Extension(
    "cpp_podem",
    [f"src/{source}" for source in CORE_SOURCES],
    include_dirs=["src", pybind11.get_include()],
    language="c++",
)

setup(
    name="cpp-podem",
    version="0.1.0",
    description="pybind11 training bridge for the C++ PODEM engine",
    packages=(
        find_packages("python")
        + find_packages("vendor/deepgate_recgnn_extractor")
    ),
    package_dir={
        "": "python",
        "deepgate_recgnn_extractor": DEEPGATE_PACKAGE_ROOT,
    },
    python_requires=">=3.9",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExt},
)
