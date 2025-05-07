# =============================================================================
# File: analyzer.py
# Original Author: Vojtěch Vančura
# Modified by: Vojtěch Nekl
# Modified on: 3.4.2025
# Description: Provides functions to analyze and process results from a grid search.
# Notes: Modified as part of the bachelor's thesis work. Only small adjustments were made to the original code.
# =============================================================================


import pandas as pd
import os

def check_dir(dir_name):
    try:
        g = []
        for _, _, filenames in os.walk(dir_name):
            g.extend(filenames)
        #assert 'history.csv' in g
        assert 'result.csv' in g
        assert 'setup.csv' in g
        assert 'timer.csv' in g
        #assert 'val_logs.csv' in g
        return True
    except:
        return False
    

def get_data_dirs(parent_dir = "."):
    f = []
    for _, dirnames, _ in os.walk(parent_dir):
        f.extend(dirnames)

    f = [x for x in f if check_dir(os.path.join(parent_dir,x))]
    return f

def get_results(path):
    setup = pd.read_csv(os.path.join(path, "setup.csv"))
    setup = setup.set_index(setup[setup.columns[0]], drop=True)["0"].to_frame().T
    result = pd.read_csv(os.path.join(path, "result.csv"))
    result = result.set_index(result[result.columns[0]], drop=True)["0"].to_frame().T
    timer = pd.read_csv(os.path.join(path, "timer.csv"))
    timer = timer.set_index(timer[timer.columns[0]], drop=True)["0"].to_frame().T
    timer.columns=(["training_time"])
    
    ret = pd.concat([setup, timer, result], axis=1)
    
    # Try to read validation logs
    try:
        val_logs = pd.read_csv(os.path.join(path, "val_logs.csv")).iloc[:, 1:]
        if not val_logs.empty:
            last_row = val_logs.iloc[-1]  # Select the last row
            last_row.index = [f"val_{col}" for col in last_row.index]  # Rename columns
            ret = ret.assign(**last_row.to_dict())  # Add as new columns
    except FileNotFoundError:
        pass  # If val_logs.csv doesn't exist, just return the dataframe without it

    ret["dir"] = path  # Add directory path
    return ret


def get_raw_data(parent_dir=".", already_scanned=[]):
    f = get_data_dirs(parent_dir)
    #print(pd.Series([str(os.path.join(parent_dir, x)) for x in f]))
    dfs = [get_results(str(os.path.join(parent_dir, x))) for x in f if str(x) not in already_scanned]
    dfs = [df for df in dfs if not df.empty]  # Remove empty DataFrames
    data = pd.concat(dfs) if dfs else pd.DataFrame()
    
    #data.to_feather(os.path.join(parent_dir, "data.feather"))
    return data

def get_data(parent_dir="."):
    cached_data = pd.read_csv(os.path.join(parent_dir, "data.csv"))
    #print(cached_data["dir"])
    try:
        new_data = get_raw_data(parent_dir=parent_dir, already_scanned=cached_data["dir"].to_list())
        data = pd.concat([cached_data, new_data]).reset_index(drop=True)
        data.to_csv(os.path.join(parent_dir, "data.csv"), index=False)
    except ValueError:
        data = cached_data
    return data 


def check_missing(dir_name):
    """Returns True if any of the required files is missing in the dir."""
    try:
        required_files = {'result.csv', 'setup.csv', 'timer.csv'}
        present_files = set()

        for _, _, filenames in os.walk(dir_name):
            present_files.update(filenames)
            break  # Only check top-level files

        return not required_files.issubset(present_files)
    except Exception:
        return True  # If any error, assume it's missing

def get_incomplete_dirs(parent_dir="."):
    incomplete = []
    for root, dirnames, _ in os.walk(parent_dir):
        for dirname in dirnames:
            full_path = os.path.join(root, dirname)
            if check_missing(full_path):
                incomplete.append(full_path)
        break  # Only look at top-level dirs inside parent_dir
    return incomplete