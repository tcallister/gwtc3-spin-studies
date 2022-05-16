Getting started
===============

Setting up your environment
----------------------------

To make it as easy as possible to reproduce our results and/or figures, the `environment.yaml` file can be used to build a conda environment containing all the packages needed to run the code in this repository.
To set up this environment, do the following:

**Step 0**. Make sure you have conda installed. If not, see e.g. https://docs.conda.io/en/latest/miniconda.html

**Step 1**. Do the following:

.. code-block:: bash

    $ conda env create -f environment.yaml

This will create a new conda environment named *gwtc3-spin-studies*

**Step 2**. To activate the new environment, do

.. code-block:: bash

    $ conda activate gwtc3-spin-studies

You can deactivate the environment using :code:`conda deactivate`

Downloading input files and inference results
---------------------------------------------

Datafiles containing the output of our inference codes are hosted on Zenodo at https://doi.org/10.5281/zenodo.6505272, along with the input files loaded in by our inference code.
If you want to regenerate figures or rerun any of our analyses, you'll need to download this input/output data locally.
You can do this via

.. code-block:: bash

    $ cd data/
    $ . download_data_from_zenodo.sh

This script will populate the :code:`data/` directory with datafiles containing the output of our analyses.
These output files can be inspected by running the jupyter notebooks also appearing in the :code:`data/` directory; see :ref:`Effective spin models` and :ref:`Component spin models` for some additional information.
The script will also place several files in the :code:`code/input/` directory, which are needed to rerun analyses and/or regenerate figures.

Some of our figures also contain data produced by Galaudage+ (2021) [1]_ and hosted at https://github.com/shanikagalaudage/bbh_spin.
If you want to regenerate figures, you'll want to clone this repository locally (making sure to check out files stored in git-lfs) and create a link to this repository in the :code:`code/input/` folder:

.. code-block:: bash

    $ cd [.../desired/filepath/...]
    $ git clone git@github.com:shanikagalaudage/bbh_spin.git
    $ cd [.../path/to/gwtc3-spin-studies]/code/input
    $ ln -s [.../path/to/bbh_spin] galaudage-data


.. [1] `Galaudage et al., ApJL 921, L15 (2021) <https://doi.org/10.3847/2041-8213/ac2f3c>`_
