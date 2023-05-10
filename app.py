#!/usr/bin/env python
'''
This is a simple latency test of TGIS 
using a single 1024 token input script, generate 284 tokens

Port forward the TGIS to localhost as following 
oc port-forward service/tgis-onnyx-inference-server  8033:8033

requires: 
pip install protobuf grpcio-tools==1.51.1 mypy-protobuf==3.4.0 'types-protobuf>=3.20.4' --no-cache-dir

The client stub is generated using the following command:
python -m grpc_tools.protoc --proto_path=<git_clone>/fmaas-inference-server/proto --python_out=. --pyi_out=. --grpc_python_out=.   generation.proto

'''


import logging
import grpc
import generation_pb2
import generation_pb2_grpc
from google.protobuf import json_format
import timeit
import pprint
import numpy as np
import os

# Test control parameter
ITERATIONS = 100
PRINT_RESPONSE = True
SERVER_HOST_PORT = "tgis-service2-tgis.apps.sap-dat3.ibm-cpd-partnerships.com"  #"localhost:8033"
MAX_NEW_TOKENS = 248
INITIAL_WARMUP_CYCLES = int(os.environ.get('INITIAL_WARMUP_CYCLES') or '1')

if 'SERVER_HOST_PORT' in os.environ:
    SERVER_HOST_PORT = os.environ['SERVER_HOST_PORT']


# Load test script with 1024 Tokens
#with open("input_1024_token.txt", "r") as f:
#    input_script = f.read()

# Compose the TGIS raw request
request = {'model_id': 'notused',   
           'requests': [ {"text": "function helloWorld():"} ],
           'params': {  'method': 'GREEDY', 
                        'sampling': {'temperature': 0.2, # 0.00 to 1.00
                                     'top_p': 0.95, # 0.00 to 1.00,
                                    },
                        'stopping': {"maxNewTokens": MAX_NEW_TOKENS, 
                                    "min_new_tokens": MAX_NEW_TOKENS,
                                    "stop_sequences": ['<|endoftext|>']}
                    }
            }

excepted_response = {'responses': [{'generatedTokenCount': 248,
                'inputTokenCount': 1024,
                'stopReason': 'MAX_TOKENS',
                'text': '- name: SCORED | 3.2.2 | PATCH | Ensure ICMP '
                        'redirects are not accepted\n'
                        '  ansible.posix.sysctl:\n'
                        '    name: "{{ item.name }}"\n'
                        '    value: "{{ item.value }}"\n'
                        '    sysctl_set: true\n'
                        '    state: present\n'
                        '    reload: true\n'
                        '    ignoreerrors: true\n'
                        '  with_items:\n'
                        '    - name: net.ipv4.conf.all.accept_redirects\n'
                        '      value: 0\n'
                        '    - name: net.ipv4.conf.default.accept_redirects\n'
                        '      value: 0\n'
                        '  notify:\n'
                        '    - sysctl flush ipv4 route table\n'
                        '  when:\n'
                        '    - amazonlinux2cis_level1 is defined and '
                        'amazonlinux2cis_level1\n'
                        '  tags:\n'
                        '    - level1\n'
                        '    - patch\n'
                        '    - rule_3.1.1\n'
                        '    - low\n'
                        '    - patch\n'
                        '    - rule_3.1.1\n'
                        '    - low\n'
                        '    - disable_strategy\n'
                        '    - low_complexity\n'
                        '    - low_disruption'}]}

def run():
    message = json_format.ParseDict(request, generation_pb2.BatchedGenerationRequest())
    with grpc.insecure_channel(SERVER_HOST_PORT) as channel:
        stub = generation_pb2_grpc.GenerationServiceStub(channel)
        response = stub.Generate(message)
        response_dict = json_format.MessageToDict(response)
        return response_dict

latList = []

if __name__ == '__main__':
    logging.basicConfig()

    print(f"Targeting host: {SERVER_HOST_PORT}...")

    tik = timeit.default_timer()
    print(f"Starting warmup with {INITIAL_WARMUP_CYCLES} cycles...")
    # INITIAL_WARMUP_CYCLES
    for i in range(INITIAL_WARMUP_CYCLES):
        response = run()
    tok = timeit.default_timer()

    print(f"Warmup terminated in {tok-tik} sec.")

    # Start latency test 
    start_time = timeit.default_timer()
    for i in range(ITERATIONS):
        s = timeit.default_timer()
        response = run()
        latList.append(timeit.default_timer() - s)
    elapsed = timeit.default_timer() - start_time

    # print('Latency =', round(elapsed,2), 'Sec')
    if PRINT_RESPONSE:
        pprint.pprint(response)
    lat= np.array(latList)
    print('**** LATENCY TEST REPORT ****')
    print('Input Tokens: ', response['responses'][0]['inputTokenCount'])
    print('Output Tokens:', response['responses'][0]['generatedTokenCount'])
    print("Iterations: ", ITERATIONS)
    print("Min Latency:", round(np.min(lat),2), ' Sec')
    print("Max Latency:", round(np.max(lat),2), ' Sec')
    print("25 Percentile:", round(np.percentile(lat, 25),2), ' Sec')
    print("50 Percentile:", round(np.percentile(lat, 50),2), ' Sec')
    print("75 Percentile:", round(np.percentile(lat, 75),2), ' Sec')
    print("90 Percentile:", round(np.percentile(lat, 90),2), ' Sec')
    print("99 Percentile:", round(np.percentile(lat, 99),2), ' Sec')
    print("Average Latency:", round(np.average(lat),2), ' Sec')
