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

    import qiskit.qasm3 as QASM
    
    qc = QuantumCircuit()
    qc_str = QASM.dumps(qc)
    
    payload = {"circuit": qc_str}
    json_data = json.dumps(payload)
    response = requests.post('http://localhost:8085/featuresSampler', data=json_data, headers={'Content-Type': 'application/json'})
    
    if response.status_code==200:
        features = response.json()
    
        circuits_to_execute = [features["Dp1"],features["Dp2"],features["Dp3"],features["O1"],features["O2"]]
        
        """
        Backend specific circuit execution logic goes here.
        For example using QUIET's own execution api, based on AER simulator and a 14 qubit noise model
        
        payload = {"circuit": circuits_to_execute}
        json_data = json.dumps(payload)
        response = requests.post('http://localhost:8085/executeSamplerNoisyList', data=json_data, headers={'Content-Type': 'application/json'})
        if response.status_code==200:
            circuit_executions = response.json()
            features["Dp1"] = circuit_executions[0]
            features["Dp2"] = circuit_executions[1]
            features["Dp3"] = circuit_executions[2]
            features["O1"] = circuit_executions[3]
            features["O2"] = circuit_executions[4]
        
        """
        
    
    payload = features                 # after adding circuit execution results
    json_data = json.dumps(payload)
    response = requests.post('http://localhost:8085/filterSampler', data=json_data, headers={'Content-Type': 'application/json'})
    if response.status_code==200:
        filter_probs = response.json()
        
        print(f"Mitigated Probs: {filter_probs}")


### For Training new model
User can provide its own training data for a different noise model and provide the path in the configuration file

The features required for the QUIET model can be obtained by calling /featureSampler endpoint. The user can execute the feature
circuit and from the output create row entries in the training data. Each row in the training data can is a single state observed
in the circuit output. 
To train a new model run:

    python trainer.py
