from setuptools import find_packages, setup


setup(
    name="lunar_rover_tasks",
    version="0.1.0",
    extras_require={"safety": ["osqp==1.0.4"]},
    packages=find_packages(),
    package_data={"lunar_rover_tasks.tasks.multi_rover_gathering.local_goal": ["*.cu", "*.cpp", "*.h"]},
    install_requires=[
        "gymnasium>=0.29",
        "numpy>=1.26",
        "torch>=2.10",
        "pyyaml>=6.0",
    ],
)

