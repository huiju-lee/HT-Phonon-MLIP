# HT-Phonon-MLIP
This repository contains the Python scripts for journal publication on "Accelerating High-Throughput Phonon Calculations via Machine Learning Universal Potentials"

- calc_phonon.py: This python script is designed to automate phonon calculations using a machine learning interatomic potential (MLIP). It integrates Phonopy for phonon displacement generation and force constant calculations, and uses a trained ML model to calculate forces for displaced supercells. Key functionalities include generating Phonopy parameters, writing force constants, computing phonon band structures, and calculating thermal properties.
- train_model.sh: This batch job script was used for our model training.
- pk_to_xyz.py: This python script is used for converting trainining data from a pickle file format to an extended XYZ file format which is required for training.

Trained model, training data, and phonon data can be found here: https://zenodo.org/records/14262400
