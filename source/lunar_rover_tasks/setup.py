from setuptools import find_packages, setup


setup(
    name="lunar_rover_tasks",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.26",
        "torch>=2.10",
        "pyyaml>=6.0",
    ],
)

