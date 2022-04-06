# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import base64
import json
import os
import numpy as np
import requests


# Base64 encode the tensors (e.g. .jpg or .npy files) to make the input sample
# json-serializable. (Alternatively, the tensors could be loaded and converted
# to lists.)
clients_dir_abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rel_path_image = 'solution_example_clients/example_input_clients/image/003_003753.8-005152_K.jpg'
with open(os.path.join(clients_dir_abs_path, rel_path_image), 'rb') as f:
    binary_image = f.read()
encoded_image = base64.b64encode(binary_image).decode('utf-8')


# Define the input sample.
sample_dict = {
    'gi': 1.5976,
    'gk': 4.828323000000001,
    'gr': 0.4321999999999981,
    'gw1': 6.310542999999999,
    'gw2': 6.9033,
    'gz': 2.5516000000000005,
    'ij': 1.8310169999999992,
    'ik': 3.230723000000001,
    'image': encoded_image,
    'image_id': '003_003753.8-005152_K',
    'index': 3,
    'iw1': 4.712942999999999,
    'iw2': 5.3057,
    'iz': 0.9540000000000006,
    'jw1': 2.881926,
    'jw2': 3.4746830000000006,
    'kw1': 1.482219999999998,
    'kw2': 2.0749769999999987,
    'petroR50_g': 9.474340282071314,
    'petroR50_r': -0.8645234747058865,
    'ri': 1.1654000000000018,
    'rw1': 5.878343000000001,
    'rw2': 6.471100000000002,
    'rz': 2.1194000000000024,
    'ug': 0.5084000000000017,
    'ui': 2.1060000000000016,
    'uj': 3.937017000000001,
    'uk': 5.336723000000003,
    'ur': 0.9406,
    'uw1': 6.818943000000001,
    'uw2': 7.4117000000000015,
    'uz': 3.0600000000000023,
    'w1w2': 0.5927570000000006,
    'zj': 0.8770169999999986,
    'zk': 2.2767230000000005,
    'zw1': 3.758942999999999,
    'zw2': 4.351699999999999
}

# Make a request to the solution server.
url = 'http://127.0.0.1:5001/predict'
headers = {'Content-type': 'application/json'}
body = str.encode(json.dumps({"data": [sample_dict]}))
response = requests.post(url, body, headers=headers)

print(response.json())

