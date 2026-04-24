from __future__ import annotations

import sys

from setuptools import Extension, setup


extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]


setup(
    ext_modules=[
        Extension(
            "network_creation.policies._best_response_kernel",
            sources=["network_creation/policies/_best_response_kernel.c"],
            extra_compile_args=extra_compile_args,
        )
    ],
)
