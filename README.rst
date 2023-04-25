===========================
Scipion plugin for cryoCARE
===========================

.. image:: https://img.shields.io/pypi/v/scipion-em-cryocare.svg
        :target: https://pypi.python.org/pypi/scipion-em-cryocare
        :alt: PyPI release

.. image:: https://img.shields.io/pypi/l/scipion-em-cryocare.svg
        :target: https://pypi.python.org/pypi/scipion-em-cryocare
        :alt: License

.. image:: https://img.shields.io/pypi/pyversions/scipion-em-cryocare.svg
        :target: https://pypi.python.org/pypi/scipion-em-cryocare
        :alt: Supported Python versions

.. image:: https://img.shields.io/pypi/dm/scipion-em-cryocare
        :target: https://pypi.python.org/pypi/scipion-em-cryocare
        :alt: Downloads

This plugin allows to use cryoCARE_ -trains a denoising U-Net for tomographic reconstruction according to the
Noise2Noise_ training paradigm- tomography methods into Scipion framework.

============
Installation
============
The plugin can be installed in user (stable) or developer (latest, may be unstable) mode:

**1. User (stable) version:**:

.. code-block::

    scipion3 installp -p scipion-em-cryocare

**2. Developer (latest, may be unstable) version:**:

* Clone the source code repository:

.. code-block::

    git clone https://github.com/scipion-em/scipion-em-cryocare.git

* Install:

.. code-block::

    scipion3 installp -p local/path/to/scipion-em-cryocare --devel

=========
Protocols
=========
The integrated protocols are:

1. Load a previously trained model.

2. Generate the training data.

3. Training: uses two data-independent reconstructed tomograms to train a 3D cryoCARE network.

4. Predict: generates the final restored tomogram by applying the cryoCARE trained network to both
even/odd tomograms followed by per-pixel averaging.

=====
Tests
=====

The installation can be checked out running some tests. To list all of them, execute:

.. code-block::

     scipion3 tests --grep cryocare

To run all of them, execute:

.. code-block::

     scipion3 tests --grep cryocare --run

========
Tutorial
========

The test generates a cryoCARE workflow that can be used as a guide about how to use cryoCARE. The even/odd tomograms
required to use cryoCARE can be generated inside Scipion with:

1. Plugin scipion-em-motioncorr_: protocol "align tilt-series movies".

2. Plugin scipion-em-xmipptomo_: protocol "tilt-series flexalign".

==========
References
==========

* `Cryo-CARE: Content-Aware Image Restoration for Cryo-Transmission Electron Microscopy Data. <http://doi.org/10.1109/ISBI.2019.8759519>`_
  Tim-Oliver Buchholz et al., 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019).

===================
Contact information
===================

If you experiment any problem, please contact us here: scipion-users@lists.sourceforge.net or open an issue_.

We'll be pleased to help.

*Scipion Team*


.. _cryoCARE: https://github.com/juglab/cryoCARE_pip
.. _Noise2Noise: https://arxiv.org/pdf/1803.04189.pdf
.. _scipion-em-motioncorr: https://github.com/scipion-em/scipion-em-motioncorr
.. _scipion-em-xmipptomo: https://github.com/I2PC/scipion-em-xmipptomo
.. _issue: https://github.com/scipion-em/scipion-em-cryocare/issues