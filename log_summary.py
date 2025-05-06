"""
Author: Lev Kakasenko
Description:
Prints out a table summarizing relevant information from the logs.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""
import os
import pickle
from tabulate import tabulate

headers = ['heur', 'prior_type', 'reconstruction', 'data_matrix', 'noise']

# convert the dictionary of sensor placement heuristics into a string
def heur_to_string(heur_input):
    if isinstance(heur_input, dict):
        params = ", ".join([f'{key}={value}' for key, value in heur_input['params'].items()])
        heur_input = heur_input['name'] + '; ' + params
        return heur_input
    else:
        return heur_input

log_data_list = []

# print out the contents of the log files
for root, dirs, files in os.walk("logs"):
    for file in files:
        if file.endswith(".pickle"):
            with open(f'logs/{file}', 'rb') as f:
                data = pickle.load(f)
                values = list(data['metadata'][i] for i in headers)
                values[0] = heur_to_string(values[0])
                values = [file[4:12]] + values # add the file index number
                log_data_list.append(values)
headers = ['id'] + headers
print(tabulate(log_data_list, headers=headers, tablefmt="grid"))
