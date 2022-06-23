#!/bin/bash

# Download and unzip
curl https://zenodo.org/record/6555145/files/gwtc3-spin-studies-data.zip --output "gwtc3-spin-studies-data.zip"
unzip gwtc3-spin-studies-data.zip

# Move input data to ../code/input/
mv sampleDict_FAR_1_in_1_yr.pickle ../code/input/
mv injectionDict_FAR_1_in_1.pickle ../code/input/
mv posteriors_gaussian_spin_samples_FAR_1_in_1.json ../code/input/

# Remove original zip files and annoying Mac OSX files
rm gwtc3-spin-studies-data.zip
rmdir __MACOSX/
