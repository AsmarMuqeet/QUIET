import os
import pickle
from collections import defaultdict

import qiskit.qasm3 as QASM
import yaml
from lightgbm import LGBMClassifier, early_stopping, LGBMRegressor
from qiskit import transpile, QuantumCircuit
import copy
import numpy as np
import pandas as pd
from qiskit.converters import circuit_to_dag
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from qiskit.quantum_info import hellinger_distance
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2 as FakeMelbourne

def get_noise_model():
    noise_model = AerSimulator.from_backend(FakeMelbourne())
    return noise_model

def qasm_to_qiskt(qasm: str) -> str:
    qiskit_circuit = QASM.loads(qasm)
    qiskit_circuit = transpile(qiskit_circuit,basis_gates=["cx",'rz','id','sx','x'],optimization_level=3)
    return copy.deepcopy(qiskit_circuit)

def get_subcircuit_sampler(circuit):
    data = circuit.data
    qregs = circuit.qregs
    cregs = circuit.cregs
    operations = [x for x in data if x.operation.name != 'measure']
    measurements = [x for x in data if x.operation.name == 'measure']
    P1 = int(np.percentile([x for x in range(len(operations))],q=25))
    P2 = int(np.percentile([x for x in range(len(operations))], q=50))
    P3 = int(np.percentile([x for x in range(len(operations))], q=75))

    instructions = operations[0:P1+1]
    qc = QuantumCircuit()
    for reg in qregs:
        qc.add_register(reg)
    for reg in cregs:
        qc.add_register(reg)
    for inst in instructions:
        qc.append(inst)
    qc = qc.compose(qc.inverse())
    for inst in measurements:
        qc.append(inst)
    qc1 = copy.deepcopy(qc)

    instructions = operations[0:P2+1]
    qc = QuantumCircuit()
    for reg in qregs:
        qc.add_register(reg)
    for reg in cregs:
        qc.add_register(reg)
    for inst in instructions:
        qc.append(inst)
    qc = qc.compose(qc.inverse())
    for inst in measurements:
        qc.append(inst)
    qc2 = copy.deepcopy(qc)

    instructions = operations[0:P3+1]
    qc = QuantumCircuit()
    for reg in qregs:
        qc.add_register(reg)
    for reg in cregs:
        qc.add_register(reg)
    for inst in instructions:
        qc.append(inst)
    qc = qc.compose(qc.inverse())
    for inst in measurements:
        qc.append(inst)
    qc3 = copy.deepcopy(qc)

    return qc1, qc2, qc3

def get_gate_counts(dag):
    one_qubit_gates = 0
    two_qubit_gates = 0
    for node in dag.nodes():
        try:
            if node.qargs:
                if node.name == "barrier" or node.name == "measure":
                    continue
                if len(node.qargs) == 1:
                    one_qubit_gates += 1
                else:
                    two_qubit_gates += 1
        except:
            pass
    return one_qubit_gates, two_qubit_gates

def get_features_sampler(qasm: str):
    circuit = qasm_to_qiskt(qasm)
    qc1, qc2, qc3 = get_subcircuit_sampler(circuit)
    circuit_width = circuit.width()
    circuit_depth = circuit.depth()
    dag = circuit_to_dag(circuit)
    one_qubit_gates, two_qubit_gates = get_gate_counts(dag)
    data = {"width": circuit_width, "depth": circuit_depth,
            "gate_counts_1q": one_qubit_gates,"gate_counts_2q": two_qubit_gates,
            "Dp1": QASM.dumps(qc1), "Dp2": QASM.dumps(qc2), "Dp3": QASM.dumps(qc3),
            'O1':QASM.dumps(circuit),'O2':QASM.dumps(circuit)}
    return data

def execute_circuit_noise_sampler(qasm: str):
    circuit = qasm_to_qiskt(qasm)
    tcirc = transpile(circuit, noise_model)
    result_noise = noise_model.run(tcirc,shots=20000,seed_simulator=1997).result()
    counts_noise = result_noise.get_counts()

    result = {}
    for k,v in counts_noise.items():
        result[int(k.replace(" ",""),2)] = v

    return result

def execute_circuits_noise_sampler(qasm_list):
    results = []
    circuits = []
    for c in qasm_list:
        circuit = qasm_to_qiskt(c)
        tcirc = transpile(circuit, noise_model)
        circuits.append(tcirc)

    result_noise = noise_model.run(circuits,shots=20000,seed_simulator=1997).result()
    counts_noise = result_noise.get_counts()
    for c in counts_noise:
        result = {}
        for k,v in c.items():
            result[int(k.replace(" ",""),2)] = v
        results.append(result)

    return results

def execute_circuit_sampler(qasm: str):
    circuit = qasm_to_qiskt(qasm)
    circuit.remove_final_measurements()
    circuit.measure_all()
    sampler = AerSimulator()
    shots = 1024
    job = sampler.run(circuit, shots=shots)
    sampling_result = job.result().get_counts()
    results = dict([(int(k.replace(" ",""), 2), v) for k, v in sampling_result.items()])
    return results

def execute_circuits_sampler(qasm_list):
    sampler = AerSimulator()
    results = []
    for c in qasm_list:
        circuit = qasm_to_qiskt(c)
        circuit.remove_final_measurements()
        circuit.measure_all()
        shots = 1024
#         if circuit.num_qubits>4:
#             shots = 20000
        job = sampler.run(circuit, shots = shots)
        sampling_result = job.result().get_counts()
        results.append(dict([(int(k.replace(" ",""),2),v) for k,v in sampling_result.items()]))
    return results

def convert_features(features: dict):
    data = pd.DataFrame(
        columns=["width", "depth", "gate_counts_1q", "gate_counts_2q", "Dp1", "Dp2", "Dp3", "POS", "POF", "ODR","State"])
    width = features["width"]
    depth = features["depth"]
    gate_counts_1q = features["gate_counts_1q"]
    gate_counts_2q = features["gate_counts_2q"]
    Dp1 = hellinger_distance({0:sum(features["Dp1"].values())},features["Dp1"])
    Dp2 = hellinger_distance({0:sum(features["Dp2"].values())},features["Dp2"])
    Dp3 = hellinger_distance({0:sum(features["Dp3"].values())},features["Dp3"])

    total_noisy_runs = defaultdict(list)
    for d in [features["O1"],features["O2"]]:
        for key, value in d.items():
            total_noisy_runs[key].append(value / sum(d.values()))

    for each_state in total_noisy_runs:
        mean_prob = np.round(np.mean(total_noisy_runs[each_state]), 3)
        #mean_prob = np.round(np.sum(total_noisy_runs[each_state]), 3)
        #mean_prob = np.round(total_noisy_runs[each_state][0], 3)
        odds = [v / (1 - v) if v != 1 else 1 for v in total_noisy_runs[each_state]]
        mean_odds = np.round(np.mean(odds), 3)
        #mean_odds = np.round(np.sum(odds), 3)
        probf = [1 - v for v in total_noisy_runs[each_state]]
        mean_probf = np.round(np.mean(probf), 3)
        data.loc[len(data)] = {"width":width, "depth":depth, "gate_counts_1q":gate_counts_1q, "gate_counts_2q":gate_counts_2q, "Dp1":Dp1, "Dp2":Dp2, "Dp3":Dp3, "POS":mean_prob, "POF":mean_probf, "ODR":mean_odds,"State":each_state}

    return data

def convert_features_with_ideal(features: dict):
    data = pd.DataFrame(
        columns=["width", "depth", "gate_counts_1q", "gate_counts_2q", "Dp1", "Dp2", "Dp3", "POS", "POF", "ODR","target"])
    width = features["width"]
    depth = features["depth"]
    gate_counts_1q = features["gate_counts_1q"]
    gate_counts_2q = features["gate_counts_2q"]
    Dp1 = hellinger_distance({0: sum(features["Dp1"].values())}, features["Dp1"])
    Dp2 = hellinger_distance({0: sum(features["Dp2"].values())}, features["Dp2"])
    Dp3 = hellinger_distance({0: sum(features["Dp3"].values())}, features["Dp3"])

    total_noisy_runs = defaultdict(list)
    for d in [features["O1"], features["O2"]]:
        for key, value in d.items():
            total_noisy_runs[key].append(value / sum(d.values()))

    total_ideal_runs = defaultdict(list)
    for d in [features["O1_I"], features["O2_I"]]:
        for key, value in d.items():
            total_ideal_runs[key].append(value / sum(d.values()))

    for each_state in total_noisy_runs:
        if each_state in total_ideal_runs.keys():
            ideal_value = np.mean(total_ideal_runs[each_state])
        else:
            ideal_value = -1
        mean_prob = np.round(np.mean(total_noisy_runs[each_state]), 3)
        odds = [v / (1 - v) if v != 1 else 1 for v in total_noisy_runs[each_state]]
        mean_odds = np.round(np.mean(odds), 3)
        probf = [1 - v for v in total_noisy_runs[each_state]]
        mean_probf = np.round(np.mean(probf), 3)
        data.loc[len(data)] = {"width": width, "depth": depth, "gate_counts_1q": gate_counts_1q,
                               "gate_counts_2q": gate_counts_2q, "Dp1": Dp1, "Dp2": Dp2, "Dp3": Dp3, "POS": mean_prob,
                               "POF": mean_probf, "ODR": mean_odds,"target":ideal_value}

    return data


def regression_train_test_balanced_split(df, target_col, test_size=0.2, bins=10, random_state=None):
    """
    Split a regression dataset into training and testing sets, ensuring balanced splits of the target variable.

    Args:
    df (pd.DataFrame): The input dataframe.
    target_col (str): The column name of the target variable.
    test_size (float): Proportion of the dataset to include in the test split.
    bins (int): The number of bins to discretize the target variable for stratified sampling.
    random_state (int, optional): Random state for reproducibility.

    Returns:
    pd.DataFrame: Training set.
    pd.DataFrame: Test set.
    """
    # Discretize the target variable into bins
    df['binned_target'] = pd.qcut(df[target_col], q=bins, labels=False, duplicates='drop')

    # Separate features and target variable
    X = df.drop(columns=[target_col, 'binned_target'])
    y = df[target_col]
    y_binned = df['binned_target']

    # Perform a stratified split based on the binned target
    X_train, X_test, y_train, y_test, _, y_binned_test = train_test_split(
        X, y, y_binned, test_size=test_size, stratify=y_binned, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def sampler_lgbm(name,data):
    regression_data = copy.deepcopy(data)
    data.loc[data["target"] == -1, "target"] = 0
    data.loc[data["target"] != 0, "target"] = 1
    #data = data.loc[(data['target'] == 0) | (data['target'] > 0.07)]
    #regression_data = copy.deepcopy(data)
    #regression_data.loc[regression_data["target"] != 0, "target"] = 1
    y = data["target"]
    data.drop(["target"], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3, random_state=42)
    # X_train, X_test, y_train, y_test = data.loc[X_train_index], data.loc[X_test_index],y.loc[y_train_index], y.loc[y_test_index]
    # X_train, X_test, y_train, y_test = regression_train_test_balanced_split(regression_data,"target",0.35,20,1997)
    
    gbmr = LGBMClassifier()
    gbmr.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             callbacks=[early_stopping(1000)])

    preds = gbmr.predict(data)
    regression_data = regression_data.loc[preds==1]
    y = regression_data["target"]
    regression_data.drop(["target"], axis=1, inplace=True)

    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1', 'l2'],
        'n_estimators': 10000,
        'learning_rate': 0.09,
        'num_leaves': 240,
        'max_depth': 12,
        'min_data_in_leaf': 400,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'feature_fraction': 0.3,
        "verbose": -1
    }
    #reg = LGBMRegressor(**hyper_params)
    #reg.fit(regression_data,y)
    #preds = reg.predict(regression_data)
    reg = LinearRegression()
    reg.fit(regression_data[["width","POS","ODR","POF"]],y)
    preds = reg.predict(regression_data[["width","POS","ODR","POF"]])
    #reg.fit(regression_data,y)
    #preds = reg.predict(regression_data)
    print(mean_absolute_error(preds,y))

    # gbmr = LinearRegression()
    # gbmr.fit(X_train, y_train)

    # from mambular.models import MLPRegressor
    # gbmr = MLPRegressor(treat_all_integers_as_numerical=True)
    # gbmr.fit(X_train,y_train,max_epochs=100,batch_size=256,lr=1e-3,patience=8)
    
    y_pred = gbmr.predict(X_test)
    mse = accuracy_score(y_pred, y_test)
    with open(f'model/{name}', 'wb') as f:
        pickle.dump({"c":gbmr,"r":reg}, f)

    return {"satus":f"Model {name} is saved","regression_error":mse}


def data_generation():
    data = pd.DataFrame(
        columns=["width", "depth", "gate_counts_1q", "gate_counts_2q", "Dp1", "Dp2", "Dp3", "POS", "POF", "ODR"])

    for file in tqdm(os.listdir(os.path.join(os.getcwd(),"training_circuits"))):
        print(file)
        qc = QuantumCircuit.from_qasm_file(os.path.join(os.getcwd(),f"training_circuits/{file}"))
        circ_str = QASM.dumps(qc)
        features = get_features_sampler(circ_str)
        Dp1 = features["Dp1"]
        Dp2 = features["Dp2"]
        Dp3 = features["Dp3"]
        O1 = features["O1"]
        O2 = features["O2"]
        features["Dp1"], features["Dp2"], features["Dp3"], features["O1"], features[
            "O2"] = execute_circuits_noise_sampler([Dp1, Dp2, Dp3, O1, O2])
        features["O1_I"], features["O2_I"] = execute_circuits_sampler([O1, O2])
        temp = convert_features_with_ideal(features)
        data = pd.concat([data, temp], ignore_index=True)
        data.to_csv("training_data.csv", index=False)

    data = pd.read_csv("training_data.csv")
    return data
def train_sampler_model(name):
    if "TRAINING_DATA_DIR" in config.keys():
        try:
            data = pd.read_csv(config["TRAINING_DATA_DIR"])
        except:
            data = data_generation()
    else:
        data = data_generation()

    return sampler_lgbm(name,data)

def filter_df(df,model):

    clf_df = copy.deepcopy(df)
    reg = model["r"]
    clf = model["c"]
    clf_df.drop(["State"], axis=1, inplace=True)
    c_preds = clf.predict(clf_df,num_iteration=clf.best_iteration_)
    if df.loc[c_preds==1].shape[0]>0:
        df = df.loc[c_preds == 1]
        states = df["State"].values
        df.drop(["State"], axis=1, inplace=True)
        r_preds = reg.predict(df[["width" ,"POS","ODR","POF"]])
        #r_preds = reg.predict(df)
        result = {}
        for i in range(len(r_preds)):
            if float(r_preds[i])<0:
                pass
            elif float(r_preds[i])>1:
                result[int(states[i])] = 0.8
            else:
                result[int(states[i])] = float(r_preds[i])

        result = dict([(k,round(v/sum(result.values()),ndigits=4)) for k,v in result.items()])

    else:
        result = {}
        for i in range(df.shape[0]):
            if df.loc[i,"POS"] > 0.01:
                result[int(df.loc[i,"State"])] = float(df.loc[i,"POS"])

        result = dict([(k, round(v / sum(result.values()), ndigits=4)) for k, v in result.items()])
    #print(result)
    return result

file = open('configuration.yml', 'r')
config = yaml.load(file, Loader=yaml.SafeLoader)
file.close()
noise_model = get_noise_model()

gbmmodel = ""
if "MODEL_NAME" in config.keys():
    with open(f'model/{config["MODEL_NAME"]}', 'rb') as f:
        gbmmodel = pickle.load(f)
else:
    print("Model not found")

if __name__ == '__main__':
    pass