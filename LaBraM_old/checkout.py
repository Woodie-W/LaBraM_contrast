import pickle
import numpy as np
from pathlib import Path

def check_data(file_paths):
    results = []
    for path in file_paths:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            print(f"Checking file: {path}, shape: {data.shape}")
            if not isinstance(data, np.ndarray):
                results.append((path, "Invalid data type"))
                print(f"Invalid data type in file: {path}")
            elif np.isnan(data).any():
                results.append((path, "NaN values found"))
                nan_indices = np.argwhere(np.isnan(data))
                print(f"NaN values found in file: {path}")
                print(f"NaN indices: {nan_indices}")
            elif np.isinf(data).any():
                results.append((path, "Infinite values found"))
                inf_indices = np.argwhere(np.isinf(data))
                print(f"Infinite values found in file: {path}")
                print(f"Infinite indices: {inf_indices}")
            else:
                results.append((path, "Data is valid"))
                print(f"Data in file {path} is valid.")
    return results

def read_error_file(error_file_path):
    file_paths = []
    with open(error_file_path, 'r') as file:
        for line in file:
            if 'file:' in line:
                files = line.split('file:')[1].strip().strip("[]").replace("'", "").split(', ')
                file_paths.extend(files)
    return file_paths

# Path to the error file
error_file_path = '/data1/wangkuiyu/model_update_code/LEM/LaBraM/checkpoints/vqnsp/error.txt'  # Update this path accordingly

# Read the file paths from the error file
file_paths = read_error_file(error_file_path)

# Check the data in these files
results = check_data(file_paths)

# Print summary
for path, status in results:
    print(f"{path}: {status}")

# Optionally, write the results to a file
output_file_path = 'error_check_results.txt'
with open(output_file_path, 'w') as file:
    for path, status in results:
        file.write(f"{path}: {status}\n")

print(f"Results written to {output_file_path}")
