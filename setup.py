from setuptools import setup, find_packages
import unittest 

setup(
    name="QFpy",  # Name of the package
    version="0.1",  # Version of the package
    author="Carlo Musolino",  # Author of the package
    author_email="carlo.musolino@gmail.com",  # Author's email
    description="QFpy is a toolbox containing learning and computing tools for quantitative finance.",  # Short description
    long_description=open('README.md').read(),  # Long description from README.md
    long_description_content_type='text/markdown',  # Format of the long description
    url="",  # URL of the project
    packages=find_packages(where="src"),  # Packages to include
    package_dir={"": "src"},  # Root directory of the packages
    classifiers=[  # Additional metadata
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Python version requirement
    install_requires=[  # List of dependencies
        "requests",
        "numpy",
        "h5py",
        "argparse",
        "tqdm",
        "pyparsing",
        "tox",
        "scipy",
        "nbval",
        "jupyter",
        "matplotlib",
        "seaborn"
    ],
    entry_points={  # Command-line tools
        'distutils.commands': [
            'test = unittest:TestProgram',
        ],
    },
    test_suite='setup.discover_tests'
)
