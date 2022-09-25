Taping Module
=============
Module: :mod:`pyradise.data.taping`

.. module:: pyradise.data.taping
    :noindex:

General
-------

The :mod:`~pyradise.data.taping` module provides the abstract :class:`Tape` mechanism for recording and playback
arbitrary data and an implementation for recording and playback transformations applied to images (see
:class:`TransformTape`).

Class Overview
--------------

The following classes are provided by the :mod:`~pyradise.data.taping` module:

+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| Class                    | Description                                                                                                                       |
+==========================+===================================================================================================================================+
| :class:`Tape`            | Base class for all :class:`Tape` subclasses.                                                                                      |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| :class:`TransformTape`   | :class:`Tape` class to record physical property changes and transformations of an :class:`~pyradise.data.image.Image` instance.   |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------+
| :class:`TransformInfo`   | Class to store and handle information about the modification / transformation of an :class:`~pyradise.data.image.Image` instance. |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------+

Details
-------

.. automodule:: pyradise.data.taping
    :show-inheritance:
    :members:
    :inherited-members: