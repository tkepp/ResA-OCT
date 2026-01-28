#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collections of static methods to read/write/handle files.

"""
import yaml
import os

def read_yaml_config(file_path):
    """
    Method to read yaml config file.

    :param file_path: path to yaml config file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")

def compute_paths(data_path:str = None, patients_list:list = None, subsmpl_factor:int = 1):
    out = []

    for patient in patients_list:
        tmp = os.path.join(data_path, patient)

        if subsmpl_factor in [8,16,32]:
            out.append(os.path.join(tmp, 'OS', f'oct_flat_subsmpl{subsmpl_factor}_os.nii.gz'))
            out.append(os.path.join(tmp, 'OD', f'oct_flat_subsmpl{subsmpl_factor}_od.nii.gz'))
        elif subsmpl_factor ==1 :
            out.append(os.path.join(tmp, 'OS', f'oct_flat_os.nii.gz'))
            out.append(os.path.join(tmp, 'OD', f'oct_flat_od.nii.gz'))
        else:
            raise NotImplementedError("subsmpl_factor not available!")
    return out