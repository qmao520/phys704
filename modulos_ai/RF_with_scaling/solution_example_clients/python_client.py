# (C) Modulos AG (2019-2022). You may use and modify this code according
# to the Modulos AG Terms and Conditions. Please retain this header.
import base64
import json
import os
import numpy as np
import requests


# Define the input sample.
sample_dict = {
    'gi': 2.591599999999996,
    'gk': 4.509974,
    'gr': 1.5303000000000004,
    'gw1': 3.851807000000001,
    'gw2': 3.810811999999999,
    'gz': 3.0145999999999984,
    'ij': 0.9612500000000032,
    'ik': 1.918374000000004,
    'iw1': 1.2602070000000047,
    'iw2': 1.2192120000000024,
    'iz': 0.4230000000000018,
    'jw1': 0.2989570000000015,
    'jw2': 0.2579619999999991,
    'kw1': -0.6581669999999988,
    'kw2': -0.6991620000000012,
    'ri': 1.0612999999999957,
    'rw1': 2.3215070000000004,
    'rw2': 2.280511999999998,
    'rz': 1.4842999999999975,
    'sample_ids_generated': '2',
    'ug': -0.0636000000000009,
    'ui': 2.527999999999995,
    'uj': 3.4892499999999984,
    'uk': 4.446373999999999,
    'ur': 1.4666999999999994,
    'uw1': 3.788207,
    'uw2': 3.7472119999999975,
    'uz': 2.950999999999997,
    'w1w2': -0.0409950000000023,
    'zj': 0.5382500000000014,
    'zk': 1.4953740000000018,
    'zw1': 0.8372070000000029,
    'zw2': 0.7962120000000006
}

# Make a request to the solution server.
url = 'http://127.0.0.1:5001/predict'
headers = {'Content-type': 'application/json'}
body = str.encode(json.dumps({"data": [sample_dict]}))
response = requests.post(url, body, headers=headers)

print(response.json())

