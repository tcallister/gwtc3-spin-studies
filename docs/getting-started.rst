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

Datafiles containing the output of our inference codes are hosted on Zenodo, as are the input files loaded in by our inference code.
If you want to regenerate figures or rerun any of our analyses, you'll need to download this input/output data locally.
You can do this via

.. code-block:: bash

    $ XXXXX

Some of our figures also contain data produced by [Galaudage]_ and hosted at

.. [Galaudage] test
