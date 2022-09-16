Installation
============

Install PyRaDiSe using pip (e.g. within a
`Python virtual environment <https://www.geeksforgeeks.org/python-virtual-environment/>`_):

.. code-block:: bash

   pip install pyradise

Alternatively, you can download or clone the code from `GitHub <https://github.com/ubern-mia/pyradise>`_ and install
PyRaDiSe by:

.. code-block:: bash

    git clone https://github.com/ubern-mia/pyradise
    cd pyradise
    python setup.py install

Dependencies
------------
PyRaDiSe requires Python 3.8 (or higher) and depends on the following packages:

 - `Pydicom <https://github.com/pydicom/pydicom>`_
 - `SimpleITK <https://simpleitk.org/>`_
 - `ITK <https://itk.org/>`_
 - `NumPy <https://numpy.org/>`_
 - `SciPy <https://www.scipy.org/>`_
 - `OpenCV <https://github.com/opencv/opencv-python>`_
 - `pymia <https://pymia.readthedocs.io/en/latest/>`_

Building the documentation
--------------------------
Building the documentation requires the following packages:
 - `Sphinx <https://www.sphinx-doc.org/en/master/>`_
 - `PyData Sphinx Theme <https://pydata-sphinx-theme.readthedocs.io/en/stable/>`_
 - `nbsphinx <https://nbsphinx.readthedocs.io/en/latest/>`_
 - `Sphinx-copybutton <https://sphinx-copybutton.readthedocs.io/en/latest/>`_
 - `sphinx-toolbox <https://sphinx-toolbox.readthedocs.io/en/latest/>`_
 - `Pandoc <https://pandoc.org/>`_

Install the required packages using pip:

.. code-block:: bash

   pip install -r docs/requirements.txt

.. note::
   The `Pandoc <https://pandoc.org/>`_ package needs to be installed on the machine manually.

Then, build the documentation by running:

.. code-block:: bash

   make html

The documentation will then be generated in the ``./_docs_build`` directory.

