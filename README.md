## DieselWolf: An Open-Source Data Set for Radio Frequency Deep Learning Research

There are two dataset types: DigitalModulationDataset which does not assume time synchronization and does not return the symbols of a 
message, and DigitalDemodulationDataset, which assumes that sampling begins the instant the first symbol begins (perfect time sync).
Also, DigitalDemodulationDataset returns the message symbols corresponding to the i,q signal amplitudes contained in the baseband data.

Python files contained in repo:
	
    Dataset/DigitalModulations.py - Main file containing both modulation and demodulation dataset classes.
    Dataset/ModulationTypes.py - Functions for generating data for each modulation type.
    Dataset/TransformsRF.py - PyTorch transforms for physical channel model.
    Dataset/FilterUtils.py - Filtering utility functions.
    Dataset/models.py - Model definition for hybrid multitask model and a basic ResNet.

Three Jupyter notebooks with examples showing dataset and models in use.

    Dataset/Dataset_Test_Notebook: Simple test notebook to ensure dataset and channel are working properly.
    Modulation_Classification: Training a hybrid model to perform automatic modulation recognition.
    Demodulation: Training a model to demodulate a signal.

