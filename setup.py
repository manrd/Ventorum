# Author: Manuel Alejandro Rodriguez Diaz, PhD
"""Build the optional compiled extensions of Ventorum.

The project metadata is in pyproject.toml. This file only declares the
Cython kernels. The extensions are optional: if no C compiler is available,
the installation continues and Ventorum uses its Numba and numpy kernels.
"""

import os
import subprocess
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def _macos_libomp_prefix() -> str | None:
    """Return the libomp prefix on macOS, or None when it is not found.

    A valid prefix has ``include/omp.h`` and ``lib/libomp.dylib``.
    """
    def present(prefix: str) -> bool:
        return (os.path.isfile(os.path.join(prefix, "include", "omp.h"))
                and os.path.isfile(os.path.join(prefix, "lib", "libomp.dylib")))

    env_prefix = os.environ.get("VENTORUM_LIBOMP_PREFIX")
    if env_prefix and present(env_prefix):
        return env_prefix
    try:
        out = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, timeout=60)
        brew_prefix = out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        brew_prefix = ""
    if brew_prefix and present(brew_prefix):
        return brew_prefix
    for prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        if present(prefix):
            return prefix
    return None


def _openmp_args() -> tuple[list[str], list[str]]:
    """Return (compile, link) arguments that enable OpenMP on this platform."""
    if sys.platform.startswith("linux"):
        return ["-fopenmp"], ["-fopenmp"]
    if sys.platform == "win32":
        return ["/openmp"], []
    if sys.platform == "darwin":
        # PyTorch (a dependency) ships its own libomp. Two OpenMP runtimes in
        # one process stop it ("OMP: Error #15"). The default macOS build has
        # no OpenMP; Ventorum runs the Cython kernels on Python threads instead
        # (ventorum.aero.vortex._cy_call). VENTORUM_MACOS_OPENMP=1 keeps the
        # libomp build for experiments in a process without PyTorch.
        # See docs/user/performance_limits.md.
        if os.environ.get("VENTORUM_MACOS_OPENMP", "") != "1":
            return [], []
        prefix = _macos_libomp_prefix()
        if prefix is None:
            return [], []
        return (["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
                [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"])
    return [], []


extensions = [
    Extension(
        "ventorum.legacy.aero.cython_kernels",
        ["ventorum/legacy/aero/cython_kernels.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        optional=True,
    ),
]

cy_compile, cy_link = _openmp_args()
extensions.append(
    Extension(
        "ventorum.aero.vortex_cython",
        ["ventorum/aero/vortex_cython.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=cy_compile,
        extra_link_args=cy_link,
        optional=True,
    )
)

if not cy_compile:
    if sys.platform == "darwin" and os.environ.get("VENTORUM_MACOS_OPENMP", "") != "1":
        print("macOS: ventorum.aero.vortex_cython is built without OpenMP, because PyTorch has its own "
              "libomp. Ventorum runs the Cython kernels on Python threads.")
    else:
        print("OpenMP not found: ventorum.aero.vortex_cython is built without OpenMP. Ventorum runs its "
              "kernels on Python threads. On macOS, VENTORUM_MACOS_OPENMP=1 with libomp ('brew install libomp', "
              "or VENTORUM_LIBOMP_PREFIX) enables OpenMP; it fails in a process that also loads PyTorch.")

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}, quiet=True))
