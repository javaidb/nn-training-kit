from setuptools import setup, find_packages

def get_requirements():
    with open("requirements.txt", "r") as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="nn_training_kit",
    version="1.0.1",
    packages=find_packages(),
    install_requires=get_requirements(),
) 