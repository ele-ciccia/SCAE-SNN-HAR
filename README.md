## Radar-based HAR with SNNs
This repository contains the code to reproduce the experiments presented in the article "Sparse Spike Encoding of Channel Responses for Energy Efficient Human Activity Recognition".

### Dataset

This work uses the DISC dataset available at https://ieee-dataport.org/documents/disc-dataset-integrated-sensing-and-communication-mmwave-systems (download the folder `disc_a.tar.gz`).

Create a *data* folder in the main level. Inside it, extract the downloaded .tar folder and then rename it as `raw_data`.

After that, run the dataset.py file to create the dataset, saving the files in a folder named *processed_data*.

### Repository structure

src/    -> contains modules with training function, network architectures, and other utils.
dataset.py  ->  create CIR samples from DISC dataset.
