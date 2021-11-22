from setuptools import setup, find_packages

# TODO: FIX VERSION VARIABLE PROPERLY
# Model package name
NAME = "sg_custom_catboost"
# Current Version
VERSION = "0.0.1"

# Dependecies for the package
with open("requirements.txt") as r:
    DEPENDENCIES = [
        dep
        for dep in map(str.strip, r.readlines())
        if all([not dep.startswith("#"), not dep.endswith("#dev"), len(dep) > 0])
    ]

# Project descrpition
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name=NAME,
    version=VERSION,
    description="A custon containerized model for SageMaker.",
    long_description=LONG_DESCRIPTION,
    author="Victor Rabêllo",
    packages=find_packages(exclude=("tests", "docs")),
    entry_points={
        "console_scripts": ["{name}={name}.main:cli".format(name=NAME)],
    },
    # external packages as dependencies
    install_requires=DEPENDENCIES,
)
