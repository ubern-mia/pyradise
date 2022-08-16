# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import shutil
import sys

basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, basedir)

# -- Copy example Jupyter notebooks for documentation building
shutil.copyfile(os.path.join(basedir, 'examples', 'conversion', 'dicom_to_nii.ipynb'),
                os.path.join(basedir, 'docs', 'examples.conversion.dicom_to_nii.ipynb'))

shutil.copyfile(os.path.join(basedir, 'examples', 'conversion', 'nii_to_dicom.ipynb'),
                os.path.join(basedir, 'docs', 'examples.conversion.nii_to_dicom.ipynb'))

shutil.copyfile(os.path.join(basedir, 'examples', 'inference', 'basic.ipynb'),
                os.path.join(basedir, 'docs', 'examples.inference.basic.ipynb'))

shutil.copyfile(os.path.join(basedir, 'examples', 'inference', 'container.ipynb'),
                os.path.join(basedir, 'docs', 'examples.inference.container.ipynb'))

shutil.copyfile(os.path.join(basedir, 'examples', 'conversion', 'modality_config_generation.ipynb'),
                os.path.join(basedir, 'docs', 'examples.conversion.modality_config_generation.ipynb'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyRaDiSe'
copyright = '2022, Elias Ruefenacht, ARTORG Center for Biomedical Engineering Research, University of Bern, Switzerland'
author = 'Elias Ruefenacht'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.coverage',
              'sphinx.ext.githubpages',
              'sphinx.ext.imgmath',
              'sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'sphinx_copybutton',
              'sphinxemoji.sphinxemoji',
              'sphinx_toolbox.more_autodoc.autonamedtuple',
              'sphinx_autodoc_typehints']

napoleon_use_ivar = True
autodoc_member_order = 'bysource'
add_module_names = False

templates_path = ['_templates']
exclude_patterns = []

# -- Options for nbsphinx ----------------------------------------------------
nbsphinx_allow_errors = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

