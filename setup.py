import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

ext_modules = []
if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            "groupcompress_py_ext",
            [
                "groupcompress_py_ext/groupcompress_cuda.cu",
                "groupcompress_py_ext/groupcompress_cpu.cpp",
            ],
        )
    )
else:
    ext_modules.append(
        CppExtension(
            "groupcompress_py_ext",
            [
                "groupcompress_py_ext/groupcompress_cpu.cpp",
            ],
        )
    )

setup(
    name="groupcompress_py",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "pytest-benchmark",
    ],
)
