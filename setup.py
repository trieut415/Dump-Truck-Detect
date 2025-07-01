from setuptools import setup, find_packages

setup(
    name="dumptruck_detector",
    version="0.1.0",
    description="Dumptruck Detector",
    author="Trieu Tran",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "dumptruck_detector.resources": ["*"],
    },
    # entry_points={
    #     "console_scripts": [
    #         "dumptruck = dumptruck_detector.cli:main",
    #     ],
    # },
)
