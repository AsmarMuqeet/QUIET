import warnings

import joblib
from art import tprint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

warnings.simplefilter(action='ignore', category=FutureWarning)
from collections import defaultdict
from qiskit import qasm2, QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import hellinger_distance
import numpy as np
import pandas as pd
from qiskit.converters import *
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_ibm_runtime import SamplerV2 as RSampler
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import yaml
import os
from tqdm.auto import tqdm
import lightgbm as lgb

file = open('configuration.yml', 'r')
config = yaml.load(file, Loader=yaml.SafeLoader)
file.close()

def get_noise_model():
    service = QiskitRuntimeService(channel=config['IBM_CHANNEL'], token=config['QISKIT_TOKEN'])
    backend = service.get_backend(config['IBM_COMPUTER'])
    noise_model = NoiseModel.from_backend(backend)
    return noise_model

def get_backend():
    service = QiskitRuntimeService(channel=config['IBM_CHANNEL'], token=config['QISKIT_TOKEN'])
    backend = service.get_backend(config['IBM_COMPUTER'])
    return backend

print(f"Fetching {config['IBM_COMPUTER']} Backend...")
BACKEND, NOISEMODEL = get_backend(), get_noise_model()

def execute_on_backend(qc, simulator=True):
    backend = BACKEND
    if simulator:
        noise_model = NOISEMODEL
        noisy_sampler = Sampler(default_shots=config['SHOTS'],
                                options=dict(backend_options=dict(noise_model=noise_model)))
        if isinstance(qc, list):
            qc_counts = []
            isa_circuit = [x.copy() for x in qc]
            result = noisy_sampler.run(isa_circuit).result()
            for i in range(len(isa_circuit)):
                try:
                    qc_counts.append(result[i].data.meas.get_counts())
                except:
                    qc_counts.append(result[i].data.c.get_counts())
            return qc_counts
        else:
            isa_circuit = qc.copy()
            result = noisy_sampler.run([isa_circuit]).result()
            try:
                qc_counts = result[0].data.meas.get_counts()
            except:
                qc_counts = result[0].data.c.get_counts()

            return qc_counts
    else:
        noisy_sampler = RSampler(backend=backend)
        noisy_sampler.default_shots = config['SHOTS']
        if isinstance(qc, list):
            qc_counts = []
            isa_circuit = [x.copy() for x in qc]

            result = noisy_sampler.run(isa_circuit).result()
            for i in range(len(isa_circuit)):
                try:
                    qc_counts.append(result[i].data.meas.get_counts())
                except:
                    qc_counts.append(result[i].data.c.get_counts())
            return qc_counts
        else:
            isa_circuit = qc.copy()
            result = noisy_sampler.run([isa_circuit]).result()
            try:
                qc_counts = result[0].data.meas.get_counts()
            except:
                qc_counts = result[0].data.c.get_counts()
            return qc_counts


def execute_ideal(qc):
    sampler = Sampler(default_shots=config['SHOTS'])
    if isinstance(qc, list):
        qc_counts = []
        isa_circuit = [x.copy() for x in qc]
        isa_circuit = [remove_idle_wires(x) for x in isa_circuit]
        result = sampler.run(isa_circuit).result()
        for i in range(len(isa_circuit)):
            try:
                qc_counts.append(result[i].data.meas.get_counts())
            except:
                qc_counts.append(result[i].data.c.get_counts())
        return qc_counts
    else:
        isa_circuit = qc.copy()
        isa_circuit = remove_idle_wires(isa_circuit)
        result = sampler.run([isa_circuit]).result()
        try:
            qc_counts = result[0].data.meas.get_counts()
        except:
            qc_counts = result[0].data.c.get_counts()
        return qc_counts


def load_training_circuits():
    noise_model = BACKEND
    print("Reading training circuits...")
    #pass_manager = generate_preset_pass_manager(0, noise_model)
    circuit_files = sorted(os.listdir(config['TRAINING_CIRCUIT_DIR']))
    circuits = []
    for file in tqdm(circuit_files):
        qc = qasm2.load(os.path.join(config['TRAINING_CIRCUIT_DIR'], file),
                        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        #isa_circuit = pass_manager.run(qc)
        #circuits.append(isa_circuit)
        circuits.append(qc)
    return circuits


def qubit_count_gates(qc):
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_idle_wires(qc):
    qc_out = qc.copy()
    gate_count = qubit_count_gates(qc_out)
    for qubit, count in gate_count.items():
        if count == 0:
            qc_out.qubits.remove(qubit)
    return qc_out


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



def get_dpe(counts):
    bits = len(list(counts.keys())[0])
    shots = sum(list(counts.values()))
    P = {"0" * bits: shots}
    return np.round(hellinger_distance(P, counts),3)


def get_subcircuit(circuit):
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
    qc1 = qc.copy()

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
    qc2 = qc.copy()

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
    qc3 = qc.copy()

    pass_manager = generate_preset_pass_manager(0, BACKEND)
    qc1 = remove_idle_wires(pass_manager.run(qc1))
    qc2 = remove_idle_wires(pass_manager.run(qc2))
    qc3 = remove_idle_wires(pass_manager.run(qc3))

    return qc1, qc2, qc3

def calculate_output_features(noisy_count: list, ideal_count: list = None):
    if isinstance(noisy_count, list):

        total_noisy_runs = defaultdict(list)
        for d in noisy_count:
            for key, value in d.items():
                total_noisy_runs[key].append(value / sum(d.values()))

        feature_result = {}

        for each_state in total_noisy_runs:
            median_prob = np.round(np.median(total_noisy_runs[each_state]), 3)
            odds = [v / (1 - v) if v != 1 else 1 for v in total_noisy_runs[each_state]]
            median_odds = np.round(np.median(odds), 3)
            probf = [1 - v for v in total_noisy_runs[each_state]]
            median_probf = np.round(np.median(probf), 3)
            feature_result[each_state] = {"POS": median_prob, "ODR": median_odds, "POF": median_probf,
                                          "STW": each_state.count("1")}

        if ideal_count != None:
            total_ideal_runs = defaultdict(list)
            for d in ideal_count:
                for key, value in d.items():
                    total_ideal_runs[key].append(value / sum(d.values()))

            for each_state in feature_result:
                if each_state in total_ideal_runs.keys():
                    feature_result[each_state]["ideal_prob"] = np.round(np.median(total_ideal_runs[each_state]), 3)
                else:
                    feature_result[each_state]["ideal_prob"] = 0

        return feature_result

    else:
        raise Exception("Unknow type for counts")


def get_features(circuit, use_simulator=True, ideal=False):

    qc1, qc2, qc3 = get_subcircuit(circuit)
    pass_manager = generate_preset_pass_manager(0, BACKEND)
    trans_circuit = pass_manager.run(circuit)
    trans_circuit = remove_idle_wires(trans_circuit)
    circuit_width = trans_circuit.width()
    circuit_depth = trans_circuit.depth()
    dag = circuit_to_dag(trans_circuit)
    one_qubit_gates, two_qubit_gates = get_gate_counts(dag)
    counts = execute_on_backend([trans_circuit, trans_circuit, qc1, qc2, qc3], simulator=use_simulator)
    qc_counts = counts[0:2]
    qc1_counts = counts[2]
    qc2_counts = counts[3]
    qc3_counts = counts[4]
    dp1 = get_dpe(qc1_counts)
    dp2 = get_dpe(qc2_counts)
    dp3 = get_dpe(qc3_counts)
    ideal_counts = None
    if ideal:
        ideal_counts = execute_ideal([trans_circuit, trans_circuit])
    out_features = calculate_output_features(qc_counts, ideal_counts)
    features = {}
    for each_state in out_features:
        features[each_state] = {"width":circuit_width,
                                "depth":circuit_depth,
                                "one_gates":one_qubit_gates,
                                "two_gates":two_qubit_gates,
                                "Dp1":dp1,
                                "Dp2":dp2,
                                "Dp3":dp3}

        features[each_state]["P"] = out_features[each_state]["POS"]
        features[each_state]["O"] = out_features[each_state]["ODR"]
        features[each_state]["F"] = out_features[each_state]["POF"]
        features[each_state]["S"] = out_features[each_state]["STW"]
        if ideal:
            features[each_state]["ideal_prob"] = out_features[each_state]["ideal_prob"]

    return features


def data_generation():
    data = pd.DataFrame(
        columns=["width", "depth", "one_gates", "two_gates", "Dp1", "Dp2", "Dp3", "P", "O", "F", "S", "ideal_prob"])

    circuits = load_training_circuits()
    print("Generating Training Data...")
    for circuit in tqdm(circuits):
        features = get_features(circuit,use_simulator=True,ideal=True)
        for state in features:
            data = data._append(features[state], ignore_index=True)

    data.to_csv("data/training.csv", index=False)
    print(f"data saved at data/training.csv")

def train_model():
    try:
        os.path.join(config['TRAINING_DATA_DIR'],"training.csv")
        DF = pd.read_csv(os.path.join(config['TRAINING_DATA_DIR'],"training.csv"))
    except:
        print("No training data, Please execute data_generation first.")
        return

    DF[DF["ideal_prob"]==-1] = 0

    df = DF.drop(index=list(DF[DF["ideal_prob"] == 0].sample(frac=0.6,replace=False).index))

    y = df["ideal_prob"]
    df.drop(["ideal_prob"], axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
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
    gbm = lgb.LGBMRegressor(**hyper_params)
    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            callbacks=[lgb.early_stopping(1000)])

    Y = DF["ideal_prob"]
    DF.drop(["ideal_prob"], axis=1, inplace=True)

    y_pred = gbm.predict(DF, num_iteration=gbm.best_iteration_)
    print('The mse of prediction is:', round(mean_squared_error(y_pred, Y), 5))
    joblib.dump(gbm, 'model/ML.model')
    print('Model saved to model/ML.model')


def predict(df):
    gbm = joblib.load(config["MODEL_SAVE"])
    y_pred = gbm.predict(df, num_iteration=gbm.best_iteration_)
    return y_pred

def filter_noise(qasm_circuit,use_simulator=True,use_job_id=None):
    circuit = qasm2.loads(qasm_circuit,custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    if use_simulator and use_job_id==None:
        data = pd.DataFrame(
            columns=["width", "depth", "one_gates", "two_gates", "Dp1", "Dp2", "Dp3", "P", "O", "F", "S"])
        features = get_features(circuit,use_simulator=use_simulator,ideal=False)
        for state in features:
            data = data._append(features[state], ignore_index=True)

        pred = predict(data)
        values = []
        for p in pred:
            if p<0.03:
                values.append(0)
            else:
                values.append(p)
            if p>1:
                values.append(1)
            else:
                values.append(p)
        total = sum(values)
        values = [v/total for v in values]

        result = {}
        for state,v in zip(features,values):
            if v>0:
                result[state] = float(v)
        return result





class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/filter':
            content_length = int(self.headers['Content-Length'])  # Get the size of the data
            post_data = self.rfile.read(content_length)  # Read the data from the request
            data = json.loads(post_data)  # Parse the JSON data
            try:
                qasm_circuit = data["qasm_circuit"]
                use_simulator = data["use_simulator"]
                use_job_id = data["use_job_id"]
                filtered_data = filter_noise(qasm_circuit, use_simulator, use_job_id)
                # Send response status code
                self.send_response(200)

                # Send headers
                self.send_header('Content-type', 'application/json')
                self.end_headers()

                # Send the JSON response
                self.wfile.write(json.dumps(filtered_data).encode('utf-8'))
            except:
                # Send response status code
                self.send_response(400)

    def filter(self, data):
        # Implement your filtering logic here
        # For example, let's just return the input data as is
        return data


def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler, port=8085):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting httpd server on port {port}')
    httpd.serve_forever()



def menu():
    while True:
        tprint("QUIET",font="starwars")
        print("--------------------------------------------------------")
        print("1)    Train A Model")
        print("2)    Data Generation")
        print("3)    Start Filter Server")
        print("q)    For Exit")
        choice = input("Enter your choice...")
        if choice == "1":
            train_model()
        elif choice == "2":
            data_generation()
        elif choice == "3":
            run()
        elif choice == "q":
            exit(0)
        else:
            print("Unknown choice....")