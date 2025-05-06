""" 
Author: Lev Kakasenko
Description:
Helper module that contains miscellaneous functions.

If you use this code in any form, please cite "Bridging the Gap Between Deterministic and
Probabilistic Approaches to State Estimation" by Lev Kakasenko,  Alen Alexanderian,
Mohammad Farazmand, and Arvind Krishna Saibaba (2025)
"""
import os
import random
import itertools
import pickle
import numpy as np

def generate_heur_key(heur: str, prior: str, reconstructor: str):
    """
    Generates a key that uniquely identifies a heuristic/prior/reconstructor combination.
    Used for loading in data from the logs and plotting.
    
    Parameters:
        heur (str): The sensor placement heuristic.
        prior (str): The formula used to compute the prior ('natural' or 'identity').
        reconstructor (str): The full-state reconstructor ('MAP' or 'DEIM').
    Returns:
        A unique key to identify the heuristic/prior/reconstructor combination.
    """
    return (heur, prior, reconstructor)


def heur_to_string(heur: str, reconstructor: str) -> str:
    """
    Converts a heuristic string and reconstructor string (that are used internally by the program)
    into a name to be displayed in plots.

    Parameters:
        heur (str): String representing the sensor placement heuristic.
        reconstructor (str): String representing the full-state reconstructor.
    Returns:
        String to be displayed (for external use such as in plots).
    """
    if heur == 'cpqr' or heur == 'cpqr_fair':
        if reconstructor == 'map':
            return 'Q-MAP'
        elif reconstructor == 'deim':
            return 'Q-DEIM'
    if heur == 'dopt_greedy':
        heur = 'Greedy D-'
    if heur == 'aopt_greedy':
        heur = 'Greedy A-'
    if heur == 'dopt':
        heur = 'D-'
    if heur == 'aopt':
        heur = 'A-'
    if reconstructor == 'map':
        reconstructor = 'MAP'
    if reconstructor == 'deim':
        reconstructor = 'DEIM'
    return heur + reconstructor


def check_missing_data(log, heuristics, priors, reconstructors, plot_type, mode_range, sensor_range):
    """
    Checks if any data is missing from a log.  Raises an exception if any data is missing.
    This function is useful in instances where we want to plot reconstruction error over a range
    of different mode/sensor values using plot_error.py, but some of these reconstruction errors
    haven't yet been computed using compute_error.py at points along this range.

    Parameters:
        log: The log dictionary (containing metadata and reconstruction errors) to be checked.
        heuristics: A list of sensor placement heuristics.
        priors: A list of priors (each of which is 'natural' or 'identity').
        reconstructors: A list of reconstructors ('MAP' or 'DEIM').  
                        For a given index i, we check the heuristic/prior/reconstructor 
                        combination of heuristics[i], priors[i], and reconstructors[i].
        plot_type: Specifies the x-axis against which error is plotted 
                   (set to 'error_vs_modes' or 'error_vs_sensors').
        mode_range: Range of number of modes to check (if plot_type='error_vs_modes').
        sensor_range: Range of number of sensors to check (if plot_type='error_vs_sensors').
    """
    missing_data = False
    exception_message = 'Missing the following data:'

    if plot_type == 'error_vs_modes':
        sensors = sensor_range[0]
    elif plot_type == 'error_vs_sensors':
        modes = mode_range[0]
    else:
        raise Exception('Invalid plot_type.')

    for idx in range(len(heuristics)):
        heur = heuristics[idx]
        prior = priors[idx]
        reconstructor = reconstructors[idx]
        heur_key = generate_heur_key(heur, prior, reconstructor)
        key_list = list(log[heur_key].keys())
        sensor_list = list(comb[1] for comb in key_list)

        heur_name = heur_to_string(heur, reconstructor)
        
        if heur == 'cpqr_fair':
            if plot_type == 'error_vs_modes':
                missing_modes = list(comb[0] for comb in key_list if comb[0] not in mode_range)
                if len(missing_modes) != 0:
                    missing_data = True
                    missing_modes_str = ", ".join([str(i) for i in missing_modes])
                    exception_message += f'\n   *{heur_name}:    modes=[{missing_modes_str}]'
            elif plot_type == 'error_vs_sensors':
                missing_sensors = list(comb[1] for comb in key_list if comb[1] not in sensor_range)
                if len(missing_sensors) != 0:
                    missing_data = True
                    missing_sensors_str = ", ".join([str(i) for i in missing_sensors])
                    exception_message += f'\n   *{heur_name}:    sensors=[{missing_sensors_str}]'
        else:
            if plot_type == 'error_vs_modes':
                mode_list = list(comb[0] for comb in key_list if comb[1]==sensors)
                missing_modes = list(mode for mode in mode_range if mode not in mode_list)
                if len(missing_modes) != 0:
                    missing_data = True
                    missing_modes_str = ", ".join([str(i) for i in missing_modes])
                    exception_message += f'\n   *{heur_name}:    sensors=[{sensors}], modes=[{missing_modes_str}]'
            elif plot_type == 'error_vs_sensors':
                sensor_list = list(comb[1] for comb in key_list if comb[0]==modes)
                missing_sensors = list(sensor for sensor in sensor_range if sensor not in sensor_list)
                if len(missing_sensors) != 0:
                    missing_data = True
                    missing_sensors_str = ", ".join([str(i) for i in missing_sensors])
                    exception_message += f'\n   *{heur_name}:    modes=[{modes}], sensors=[{missing_sensors_str}]'

    # Throw an exception if data is missing.
    if missing_data:
        raise Exception(exception_message)


def load_data(heuristics, reconstructors, priors, plot_type, 
              mode_range, sensor_range, num_features, num_samples, split_idx,
              data_matrix, noise, sigma_noise, seed):
    """
    Loads in data from a log file and returns this data as a dictionary.
    Certain parameters listed below are checked against the metadata of each of 
    the log files.  When a log file is found with matching metadata, that log file
    is loaded in.  This function is used in compute_error.py to generate log files.

    Parameters:
        heuristics: A list of sensor placement heuristics.
        reconstructors: A list of reconstructors (each of which is 'MAP' or 'DEIM'). 
        priors: A list of priors (each of which is 'natural' or 'identity'). 
                For a given index i, we check the heuristic/reconstructor/prior 
                combination of heuristics[i], reconstructors[i], priors[i].
        plot_type: Specifies the x-axis against which error is plotted 
                   (set to 'error_vs_modes' or 'error_vs_sensors').
        mode_range: Range of number of POD modes over which to evaluate.
        sensor_range: Range of number of sensors over which to evaluate.
        num_features: Number of features in a single data sample. 
        num_samples: Number of samples for training and testing.
        split_idx: The index at which the data matrix columns are split to generate the train
                   and test matrices.
        data_matrix: Dataset to evaluate on ('fourier' or 'turbulence').
        noise: Standard deviation of uncorrelated Gaussian noise added to the test data.
        sigma_noise: Model parameter representing a measurement noise standard deviation.
        seed: Random seed.
    Returns:
        A dictionary containing the reconstruction errors and sensor locations of different 
        heuristic/reconstructor/prior combinations (as specified in the metadata) over the 
        given mode_range (if plot_type='error_vs_modes') or sensor_range 
        (if plot_type='error_vs_sensors').
    """
    results_dict = {} # dictionary where all data from logs is stored
    for idx in range(len(heuristics)):
        heur = heuristics[idx]
        reconstructor = reconstructors[idx]
        prior = priors[idx]
        heur_key = generate_heur_key(heur, prior, reconstructor)
        results_dict[heur_key] = {}

        metadata = {'num_features': num_features, 'num_samples': num_samples, 'split_idx': split_idx,
                'heur': heur, 'reconstruction': reconstructor, 
                'data_matrix': data_matrix, 'prior_type': prior, 
                'noise': noise, 'sigma_noise': sigma_noise, 'seed': seed}
        
        key_list = [] # used to prevent duplicates with the same key

        for root, dirs, files in os.walk("logs"):
            for file in files:
                if file.endswith(".pickle"):
                    with open(f'logs/{file}', 'rb') as f:
                        data = pickle.load(f)
                        
                        if data['metadata'].items() == metadata.items():
                            print(file)

                            for key in data.keys():
                                if isinstance(key, tuple) and (key not in key_list):
                                    key_list.append(key)

                                    if heur == 'cpqr_fair':
                                        if plot_type == 'error_vs_modes':
                                            if key[0] in mode_range:
                                                data_ = data[key]
                                                results_dict[heur_key][key] = data_
                                        elif plot_type == 'error_vs_sensors':
                                            if key[1] in sensor_range:
                                                data_ = data[key]
                                                results_dict[heur_key][key] = data_
                                    elif key[0] in mode_range and key[1] in sensor_range:
                                        data_ = data[key]
                                        results_dict[heur_key][key] = data_
    
    return results_dict


def save_log(log: dict, index: int):
    """
    Saves data (contained in a dictionary) into a log file (of type .pickle).
    
    Parameters:
    -----------
    log (dict): 
        Dictionary containing tuples of type (number of modes, number of sensors) as keys
        and tuples of type (sensor locations, error mean, error variance) as values, along with 
        metadata concerning the sensor placement heuristic, reconstructor, etc. (the structure of
        this metadata can be seen in the metadata variable in the load_data function)
    index (int): 
        8-digit index of the log file to dump to.
    """
    with open(f"logs/log_{index}.pickle", "wb") as f:
        pickle.dump(log, f)
    f.close()


def check_logs(metadata, mode_range, sensor_range):
    """
    Determines if a file with the specified metadata already exists.  Returns
    the log and log index where information will be saved, total 'tasks' to be 
    performed (i.e. total number of mode/sensor combinations), and lists with
    information on the relevant mode/sensor combinations (based on mode_range,
    sensor_range, and what has already been computed).
    
    Parameters:
    -----------
    metadata: dict
        Metadata that uniquely identifies the log (with information including
        num_features, num_samples, etc.).  The structure of
        this metadata can be seen in the metadata variable in the load_data function.
    mode_range: range
        Range of number of modes to be used in computations (if not already computed).
    sensor_range: range
        Range of number of sensors to be used in computations (if not already computed).

    Returns:
    --------
    index: int
        Index of the log file where the information is saved.
    log: dict
        Log object (as uniquely identified by metadata) to which the information is saved.
    total_tasks: int
        Total number of combinations of [number of modes] and [number of sensors] to
        be examined (where each task corresponds to examining a given number of
        modes and sensors).
    mode_list: list
        Generated from modes_sensors; lists all mode counts to be examined; if computation
        results with a given number of modes are already fully accounted for in the logs,
        then that number of modes is excluded from mode_list.
    modes_sensors: list
        List of all combinations of number of modes/sensors to be examined.
    """

    log = {}
    log['metadata'] = metadata

    
    FILE_ALREADY_EXISTS = False
    EXISTING_FILE = 0
    key_list = []

    # walk through all logs to see if there's a log with the given metadata
    for root, dirs, files in os.walk("logs"):
        for file in files:
            if file.endswith(".pickle"):
                with open(f'logs/{file}', 'rb') as f:
                    data = pickle.load(f)

                    if data['metadata'].items() == metadata.items():
                        FILE_ALREADY_EXISTS = True
                        EXISTING_FILE = file
                        print('Working on ' + file)

                        # determine which combinations of number of modes/sensors
                        # need to be examined
                        for key in data.keys():
                            if isinstance(key, tuple) and (key not in key_list):
                                key_list.append(key)
    
    modes_sensors = list(itertools.product(mode_range, sensor_range)) # generates the list of tuples 
                                                                    # (num_modes, num_sensors)
    if metadata['heur'] == 'cpqr_fair':
        if len(sensor_range) == 1:
            modes_sensors = [(i, i) for i in mode_range if (i, i) not in key_list]
        elif len(mode_range) == 1:
            modes_sensors = [(i, i) for i in sensor_range if (i, i) not in key_list]
        else:
            raise Exception("Invalid variation of modes/sensors for cpqr_fair.")
    else:
        modes_sensors = [combo for combo in modes_sensors if combo not in key_list]
    total_tasks = len(modes_sensors) # used in the progress bar
    if total_tasks==0:
        raise Exception("No tasks to perform.")
    mode_list = list(set(combo[0] for combo in modes_sensors)) # generating the list of modes based on 
                                                               # modes_sensors (i.e. what hasn't been computed)

    if FILE_ALREADY_EXISTS: # load data from file into log if file already exists
        index = EXISTING_FILE[4:12]
        with open(f'logs/{EXISTING_FILE}', 'rb') as f:
            data = pickle.load(f)
            for key in key_list:
                log[key] = data[key]
            f.close()
    else: # create new file if file doesn't already exist and dump the log
        index = random.randint(10000000, 99999999)
        with open(f"logs/log_{index}.pickle", "wb") as f:
            print(f"Creating log_{index}.pickle")
            pickle.dump(log, f)
        f.close()

    print(f"Performing {total_tasks} tasks.")

    return index, log, total_tasks, mode_list, modes_sensors


def evenly_spaced_rows(arr: np.ndarray, n: int):
    """
    Return n evenly spaced rows from a 2D NumPy array.

    Parameters:
        arr (ndarray): A 2D NumPy array.
        n (int): The number of evenly spaced rows to return.

    Returns:
        selected_rows (ndarray): A new 2D NumPy array containing the evenly spaced rows.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    num_rows = arr.shape[0]

    if n >= num_rows:
        return arr  # Return the entire array if n is greater than or equal to the number of rows.

    step = num_rows // (n - 1)  # Calculate the step size between rows.
    selected_rows = arr[::step]  # Use slicing to select evenly spaced rows.

    return selected_rows
