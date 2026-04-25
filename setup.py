from __future__ import annotations

import sys

from setuptools import Extension, setup


extra_compile_args = ["/O2"] if sys.platform == "win32" else ["-O3"]


setup(
    ext_modules=[
        Extension(
            "network_creation.policies.best_response._kernel",
            sources=["network_creation/policies/best_response/_kernel.c"],
            extra_compile_args=extra_compile_args,
        )
    ],
)
