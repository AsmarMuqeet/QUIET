from art import tprint
from flask import Flask, request, jsonify
import utility as util

app = Flask("QUIET")
@app.route('/filterSampler', methods=['POST'])
def filter_method():
    data = request.get_json()
    if isinstance(util.gbmmodel,str):
        return jsonify({"error":"No train model found"}), 400

    for k in ["width", "depth", "gate_counts_1q", "gate_counts_2q", "Dp1", "Dp2", "Dp3", "O1", "O2"]:
        if k in data.keys():
            continue
        else:
            return jsonify({"error":f"Missing {k} in json"}), 400
    df = util.convert_features(data)
    result = util.filter_df(df,util.gbmmodel)
    return jsonify(result), 200


@app.route('/featuresSampler', methods=['POST'])
def features_method():
    data = request.get_json()
    circuit_qasm = data['circuit']
    features = util.get_features_sampler(circuit_qasm)
    return jsonify(features), 200

@app.route('/executeSamplerNoisy', methods=['POST'])
def execute_sampler_noise_method():
    data = request.get_json()
    circuit_qasm = data['circuit']
    counts = util.execute_circuit_noise_sampler(circuit_qasm)
    return jsonify(counts), 200

@app.route('/executeSamplerNoisyList', methods=['POST'])
def execute_sampler_noise_list_method():
    data = request.get_json()
    circuit_qasm = data['circuit']
    if isinstance(circuit_qasm,list):
        counts = util.execute_circuits_noise_sampler(circuit_qasm)
    else:
        counts = util.execute_circuits_noise_sampler([circuit_qasm])
    return jsonify(counts), 200

@app.route('/executeSamplerIdeal', methods=['POST'])
def execute_sampler_ideal_method():
    data = request.get_json()
    circuit_qasm = data['circuit']
    counts = util.execute_circuit_sampler(circuit_qasm)
    return jsonify(counts), 200

@app.route('/executeSamplerIdealList', methods=['POST'])
def execute_sampler_ideal_list_method():
    data = request.get_json()
    circuit_qasm = data['circuit']
    if isinstance(circuit_qasm,list):
        counts = util.execute_circuits_sampler(circuit_qasm)
    else:
        counts = util.execute_circuits_sampler([circuit_qasm])
    return jsonify(counts), 200


if __name__ == '__main__':
    tprint("QUIET", font="starwars")
    app.run(debug=False, host='0.0.0.0', port=8085)