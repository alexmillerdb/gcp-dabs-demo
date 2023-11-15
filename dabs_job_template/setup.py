"""
setup.py configuration script describing how to build and package this project.

This file is primarily used by the setuptools library and typically should not
be executed directly. See README.md for how to deploy, test, and run
the dabs_job_template project.
"""
from setuptools import setup, find_packages

import sys
sys.path.append('./src')

import dabs_job_template

setup(
    name="dabs_job_template",
    version=dabs_job_template.__version__,
    url="https://databricks.com",
    author="alex.miller@databricks.com",
    description="wheel file based on dabs_job_template/src",
    packages=find_packages(where='./src'),
    package_dir={'': 'src'},
    entry_points={
        "packages": [
            "main=dabs_job_template.main:main"
        ]
    },
    install_requires=[
        # Dependencies in case the output wheel file is used as a library dependency.
        # For defining dependencies, when this package is used in Databricks, see:
        # https://docs.databricks.com/dev-tools/bundles/library-dependencies.html
        "setuptools"
    ],
)
