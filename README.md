## Radar-based HAR with SNNs
This repository contains the code to reproduce the experiments presented in the article "Sparse Spike Encoding of Channel Responses for Energy Efficient Human Activity Recognition".

To install all the requirements, create a new environment and run from the terminal
```
pip install -r requirements.txt
```

### 1 - DISC Dataset for HAR

This work uses the **DISC dataset** available at https://ieee-dataport.org/documents/disc-dataset-integrated-sensing-and-communication-mmwave-systems (download the folder `disc_a.tar.gz`).

This dataset provides Channel Impulse Response (CIR) measurements from standard-compliant IEEE 802.11ay packets to validate Integrated Sensing and Communication (ISAC) methods. 
Specifically, it consists of almost 40 minutes of CIR sequences including signal reflections on **7 subjects** performing **4 different activities** (*walking*, *running*, *sitting down**, and **waving hands**). 

## 1.1 - Dataset generation

First, create a *data* folder in the main level. Inside it, extract the downloaded `.tar` folder and then rename it as `raw_data`.
Then, run the `dataset.py` file to generate the dataset. This script will read and preprocess the raw CIR measurements (see the paper for more info on the preprocessing step) and it will automatically save the preprocessed files in a folder named `processed_data`. It will also generate the split for training, validation and test dataset and save them. 
You can change all the parameters in the `config.py` file inside **src** folder.

### Repository structure

src/    -> contains modules with training function, network architectures, and other utils.

