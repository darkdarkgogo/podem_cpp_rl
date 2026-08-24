import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CORE_SOURCES = [
    "atpg.cpp",
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

class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            compile_args = ["/std:c++14", "/EHsc"]
        else:
            compile_args = ["-std=c++11"]
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
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExt},
)
