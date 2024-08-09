# QUIET
# Installation
Currently QUIET only supports Windows 11 and Qiskit

### Dependencies

Anaconda Python distribution is required [here](https://www.anaconda.com/products/distribution):

Steps:
1. Clone repository
2. cd QUIET 
3. conda env create -f environment.yml
4. conda activate quiet

## Usage:
##### Before running make sure all the required fields are filled in configuration.yml
    python main.py

![Example Image](quiet.png)

#### For filtering noise QUIET requires a qasm_circuit as input. An example is shown in testdriver.py
