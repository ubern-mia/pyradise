import os
import sys

from setuptools import find_packages, setup

if sys.version_info < (3, 8):
    sys.exit("Requires Python 3.8 or higher")

directory = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(directory, "pyradise", "__version__.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

with open(os.path.join(directory, "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()

REQUIRED_PACKAGES = [
    "pydicom",
    "numpy",
    "itk>=5.3rc4.post3",
    "SimpleITK",
    "opencv-python",
    "scipy",
    "vtk",
]

TEST_PACKAGES = [
    "tox >= 3.4.0",
    "tox-pipenv >= 1.10.1",
    "pytest >= 7.0.0",
    "pytest-cov >= 4.0.0",
    "pipenv >= 2022.10.12",
]

setup(
    name=about["__title__"],
    version=about["__version__"],
    description=about["__description__"],
    long_description=readme,
    long_description_content_type="text/markdown",
    author=about["__author__"],
    author_email=about["__author_email__"],
    url=about["__url__"],
    license=about["__license__"],
    python_requires=">=3.8",
    packages=find_packages(exclude=["docs", "examples", "test"]),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    keywords=[
        "medical image analysis",
        "deep learning",
        "auto-segmentation",
        "radiotherapy",
        "DICOM conversion",
        "DICOM data handling",
        "DICOM-RT Structure Sets",
    ],
)
