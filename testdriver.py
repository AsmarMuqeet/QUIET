import requests
import json
import util as U

circuits = U.load_training_circuits()
qasm_circuit = U.qasm2.dumps(circuits[0])
use_simulator = True
use_job_id = None

# Define the URL of the server
url = 'http://localhost:8085/filter'

# Define the data to send in the POST request
data = {
    "qasm_circuit": qasm_circuit,
    "use_simulator": use_simulator,
    "use_job_id": use_job_id
}

# Convert the data to JSON format
json_data = json.dumps(data)

# Send the POST request
response = requests.post(url, data=json_data, headers={'Content-Type': 'application/json'})

# Print the response from the server
print('Status Code:', response.status_code)
print('Response Body:', response.json())