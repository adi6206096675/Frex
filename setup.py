import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

# --- FREX HARDWARE AUTO-DETECTION ---
def check_cuda_availability():
    """Probes the system for the NVIDIA CUDA Compiler."""
    try:
        subprocess.check_output(["nvcc", "--version"])
        print("\n[Frex Build] NVIDIA CUDA Detected! Enabling GPU Support.\n")
        return True
    except (FileNotFoundError, OSError):
        print("\n[Frex Build] CUDA not found. Falling back to CPU (OpenMP) mode.\n")
        return False

USE_CUDA = check_cuda_availability()

class BuildExt(build_ext):
    def build_extensions(self):
        opts = ["-O3", "-std=c++17", "-fopenmp"]
        if sys.platform == "darwin":
            opts.append("-stylesheet=c++17")
        
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = ["-fopenmp"]
            ext.include_dirs = [pybind11.get_include()]
            
            # Inject a C++ Macro if CUDA is found
            if USE_CUDA:
                ext.define_macros.append(("FREX_USE_CUDA", "1"))
                
        super().build_extensions()

ext_modules = [
    Extension(
        "frex._C",
        sources=["csrc/frex_core.cpp", "csrc/bindings.cpp"],
        depends=["csrc/frex_core.hpp"],
    ),
]

setup(
    name="frex",
    version="0.1.0",
    packages=["frex"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)