from setuptools import find_packages, setup


setup(
    name="cann_bench",
    version="1.0.0",
    description="Triton Ascend operators for CANN Bench",
    packages=find_packages(),
    python_requires=">=3.10",
)
