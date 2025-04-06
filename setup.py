from setuptools import setup, find_packages

setup(
    name="nn-training-kit",
    version="1.0.0",
    packages=find_packages(include=["nn_training_kit", "nn_training_kit.*"]),
    install_requires=[
        "torch",
        "pytorch-lightning",
        "pyyaml",
        "pydantic",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    python_requires=">=3.8",
) 