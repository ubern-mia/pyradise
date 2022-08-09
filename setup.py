import os
import sys
from setuptools import find_packages, setup

if sys.version_info < (3, 7):
    sys.exit("Requires Python 3.7 or higher")

directory = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(directory, 'pyradise', '__version__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

with open(os.path.join('README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

REQUIRED_PACKAGES = [
    'pydicom >= 2.2.0',
    'numpy >= 1.12.1',
    'SimpleITK >= 2.1.1',
    'itk >= 5.2.1.post1',
    'pymia >= 0.3.2'
]

TEST_PACKAGES = [
    'pytest >= 5',
    'pytest-cov >= 2.10.0'
]

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__author_email__'],
    url=about['__url__'],
    license=about['__license__'],
    python_requires='>=3.7',
    packages=find_packages(exclude=['docs', 'examples', 'test']),
    install_requires=REQUIRED_PACKAGES,
    tests_require=REQUIRED_PACKAGES + TEST_PACKAGES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries'
    ],
    keywords=[
        'medical image analysis',
        'deep learning',
        'data handling',
        'data conversion',
        'data curation',
        'DICOM',
        'DICOM-RT',
        'RTSS'
    ]
)