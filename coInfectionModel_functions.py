###############################################################################
#
# author: Dr. Sandra Timme 
# Copyright 2023 by Dr. Sandra Timme
# 
# Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
# https://www.leibniz-hki.de/en/applied-systems-biology.html 
# HKI-Center for Systems Biology of Infection
# 
# Leibniz Institute for Natural Product Research and Infection Biology -
#   Hans Knöll Insitute (HKI)
# Adolf-Reichwein-Straße 23, 07745 Jena, Germany
# 
# Licence: BSD-3-Clause, see ./LICENSE or 
# https://opensource.org/licenses/BSD-3-Clause for full details
# 
###############################################################################

from scipy import optimize
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from itertools import repeat
import re
import multiprocessing
from numbalsoda import lsoda_sig, lsoda
import numba as nb
from numba import njit, cfunc, types
import subprocess
from operator import add
import os
import json
import sys
from contextlib import contextmanager
from scipy.stats import qmc

import logging
logger = logging.getLogger("numba");
logger.setLevel(logging.ERROR)

simulated_annealing_iterations = 0

def get_experimental_data(exp_data_path, strain_type, exp_type, condition):
    exp_data_kinetics = pd.DataFrame()
    if strain_type == "LAB" or strain_type == "MIX":
        if exp_type == "CFCM":
            exp_data_kinetics = get_experimental_data_LAB__supernatant(exp_data_path, condition)
        elif exp_type == "CoCultivation":
            exp_data_kinetics = get_experimental_data_LAB_cocultivation(exp_data_path, condition)
        else:
            print("Cannot read experimental data! Unkown exp_type: " + exp_type)
    elif strain_type == "CLIN":
        if exp_type == "CFCM":
            exp_data_kinetics = get_experimental_data_CLIN_supernatant(exp_data_path, condition)
        elif exp_type == "CoCultivation":
            exp_data_kinetics = get_experimental_data_CLIN_cocultivation(exp_data_path, condition)
        else:
            print("Cannot read experimental data! Unkown exp_type: " + exp_type)
    else:
        print("Cannot read experimental data! Unkown strain_type: " + strain_type)
    
    exp_data_kinetics['data'] = "exp"
    exp_data_kinetics['data_type'] = "population"
    exp_data_kinetics.rename(columns={'A':'value'}, inplace=True)
    exp_data_kinetics.dropna(subset = ["value"], inplace=True)

    return(exp_data_kinetics)
    
def get_experimental_data_LAB__supernatant(exp_data_path, condition):
    files = os.listdir(exp_data_path)
    exp_data_files = pd.DataFrame()
    tmp_data = list()
    for file in files:
        tmp1 = file.split("_")
        tmp2 = tmp1[1].split("-")
        strain = tmp2[0]
        # print(strain)
        supernatant = tmp2[1].removeprefix("S")
        # print(supernatant)
        if condition == "Baseline":
            tmp2 = tmp1[2].split("-")[1]
            sample = tmp2.removesuffix(".txt")
            # print(sample)
            tmp_data.append({"file_name": file, "strain": strain, "supernatant": supernatant, "sample": sample})    
        elif condition == "Mutations":
            mutation = tmp1[2]
            tmp2 = tmp1[3].split("-")[1]
            sample = tmp2.removesuffix(".txt")
            tmp_data.append({"file_name": file, "strain": strain, "supernatant": supernatant, "sample": sample, "mutation": mutation})
        elif condition == "Agr":
            mutation = tmp1[2]
            tmp2 = tmp1[3].split("-")[1]
            sample = tmp2.removesuffix(".txt")
            tmp_data.append({"file_name": file, "strain": strain, "supernatant": supernatant, "sample": sample, "agr": mutation})
        else:
            print("Unknown condition: " + condition)

    if condition == "Baseline":
        exp_data_files = pd.DataFrame(data = tmp_data, columns=("file_name", "strain", "supernatant", "sample"))   
    elif condition == "Mutations":
        exp_data_files = pd.DataFrame(data = tmp_data, columns=("file_name", "strain", "supernatant", "sample", "mutation"))
    elif condition == "Agr":
        exp_data_files = pd.DataFrame(data = tmp_data, columns=("file_name", "strain", "supernatant", "sample", "agr"))
    
    exp_data_kinetics = pd.DataFrame()
    for index, row in exp_data_files.iterrows():
        curr_exp_data = pd.read_csv(os.path.join(exp_data_path,row["file_name"]), sep=",")
        curr_exp_data['file_name'] = row["file_name"]
        curr_exp_data['strain'] = row["strain"]
        curr_exp_data['supernatant'] = row["supernatant"]
        curr_exp_data['sample'] = row["sample"]
        if condition == "Mutations":
            curr_exp_data['mutation'] = row["mutation"]
        if condition == "Agr":
            curr_exp_data['agr'] = row["agr"]
        exp_data_kinetics = pd.concat((exp_data_kinetics, curr_exp_data), ignore_index=True)
    exp_data_kinetics = exp_data_kinetics.replace("USA", "USA300")
    return(exp_data_kinetics)

def get_experimental_data_CLIN_supernatant(exp_data_path, condition):
    files = os.listdir(exp_data_path)
    exp_data_files = pd.DataFrame()
    tmp_data = list()
    for file in files:
        tmp1 = file.split("_")
        tmp2 = tmp1[1].split("-")
        strain = tmp2[0]
        supernatant = tmp2[1].removeprefix("S")
        pair = tmp1[2].split("-")[1]
        tmp2 = tmp1[3].split("-")[1]
        sample = tmp2.removesuffix(".txt")
        tmp_data.append({"file_name": file, "strain": strain, "supernatant": supernatant, "pair": pair, "sample": sample})    

    exp_data_files = pd.DataFrame(data = tmp_data, columns=("file_name", "strain", "supernatant", "pair", "sample"))   
    
    exp_data_kinetics = pd.DataFrame()
    for index, row in exp_data_files.iterrows():
        curr_exp_data = pd.read_csv(os.path.join(exp_data_path,row["file_name"]), sep=",")
        curr_exp_data['file_name'] = row["file_name"]
        curr_exp_data['strain'] = row["strain"]
        curr_exp_data['supernatant'] = row["supernatant"]
        curr_exp_data['pair'] = row["pair"]
        curr_exp_data['sample'] = row["sample"]
        exp_data_kinetics = pd.concat((exp_data_kinetics, curr_exp_data), ignore_index=True)
    exp_data_kinetics = exp_data_kinetics.replace("USA", "USA300")
    return(exp_data_kinetics)

def get_experimental_data_LAB_cocultivation(exp_data_path, condition):
    print("Read exprimental data from cocultivation experiment.")
    files = os.listdir(exp_data_path)
    tmp_data = list()
    for file in files:
        tmp1 = file.split("_")
        tmp2 = tmp1[1].split("-")
        strain1 = tmp2[0]
        strain2 = tmp2[1]
        tmp2 = tmp1[2].split("-")[1]
        sample = tmp2.removesuffix(".txt")
        if((strain2 != "TSB") & ((strain1 in ["LS1", "USA", "USA300"]) or "SA" in strain1)):
            tmp_data.append({"file_name": file, "strain1": strain2, "strain2": strain1, "sample": sample})    
        else:
            tmp_data.append({"file_name": file, "strain1": strain1, "strain2": strain2, "sample": sample})    
        # print("strain1 = " + strain1 + ", strain2 = " + strain2 + ", sample = " + sample)

    exp_data_files = pd.DataFrame(data = tmp_data, columns=("file_name", "strain1", "strain2", "sample"))
    
    exp_data_kinetics = pd.DataFrame()
    for index, row in exp_data_files.iterrows():
        curr_exp_data = pd.read_csv(os.path.join(exp_data_path,row["file_name"]), sep=",")
        curr_exp_data['file_name'] = row["file_name"]
        curr_exp_data['strain1'] = row["strain1"]
        curr_exp_data['strain2'] = row["strain2"]
        curr_exp_data['sample'] = row["sample"]
        exp_data_kinetics = pd.concat((exp_data_kinetics, curr_exp_data), ignore_index=True)
    exp_data_kinetics = exp_data_kinetics.replace("USA", "USA300")
    return(exp_data_kinetics)

def get_experimental_data_CLIN_cocultivation(exp_data_path, condition):
    print("Read exprimental data from cocultivation experiment.")
    files = os.listdir(exp_data_path)
    tmp_data = list()
    for file in files:
        tmp1 = file.split("_")
        tmp2 = tmp1[1].split("-")
        strain1 = tmp2[0]
        strain2 = tmp2[1]
        pair = tmp1[2].split("-")[1]
        tmp2 = tmp1[3].split("-")[1]
        sample = tmp2.removesuffix(".txt")
        tmp_data.append({"file_name": file, "strain1": strain1, "strain2": strain2, "pair": pair, "sample": sample})    

    exp_data_files = pd.DataFrame(data = tmp_data, columns=("file_name", "strain1", "strain2", "pair", "sample"))
    
    exp_data_kinetics = pd.DataFrame()
    for index, row in exp_data_files.iterrows():
        curr_exp_data = pd.read_csv(os.path.join(exp_data_path,row["file_name"]), sep=",")
        curr_exp_data['file_name'] = row["file_name"]
        curr_exp_data['strain1'] = row["strain1"]
        curr_exp_data['strain2'] = row["strain2"]
        curr_exp_data['pair'] = row["pair"]
        curr_exp_data['sample'] = row["sample"]
        exp_data_kinetics = pd.concat((exp_data_kinetics, curr_exp_data), ignore_index=True)
    exp_data_kinetics = exp_data_kinetics.replace("USA", "USA300")
    return(exp_data_kinetics)

def get_exp_data_CFU_counts(strain_type, exp_type, CFU_data_path, timepoint):
    exp_data_cfu = 0
    if os.path.isfile(CFU_data_path):
        exp_data_cfu = pd.read_csv(CFU_data_path, sep=",")
        colNames = list(exp_data_cfu.columns.values.tolist())
        colNames = [s.replace('.', '_') for s in colNames]
        exp_data_cfu.columns = colNames
        if strain_type == "CLIN":
            exp_data_cfu = exp_data_cfu.query("strain_type == 'Clinical'")
            exp_data_cfu['pair'] = exp_data_cfu.strain1.str.extract(r'(\d)', expand=False)
            exp_data_cfu['strain1'] = exp_data_cfu['strain1'].str.replace('[0-9]', '', regex=True)
            exp_data_cfu['strain2'] = exp_data_cfu['strain2'].str.replace('[0-9]', '', regex=True)
            exp_data_cfu['counted_strain'] = exp_data_cfu['counted_strain'].str.replace('[0-9]', '', regex=True)
        exp_data_cfu["timepoint"] = timepoint
    else: print("File path does not exist for CFU data at time 24h does not exist! " + CFU_data_path)

    return(exp_data_cfu)

def get_CFU_fractions(key, species, strain_type, exp_type, CFU_data_path, CFU_per_sample, timepoint):
    exp_data_cfu = get_exp_data_CFU_counts(strain_type, exp_type, CFU_data_path, timepoint)

    # print("exp_data_cfu at timepoint " + str(timepoint) + "h")
    # display(exp_data_cfu)
    
    strain1 = key[0]
    strain2 = key[1]
    sample = key[2]
    pair = ""
    
    cfu_fractions = pd.DataFrame()
    if strain_type == "CLIN":
        sample = key[3]
        pair = key[2]
        
    query_base = f"strain1 == '{strain1}' and strain2 == '{strain2}'"
    tmp = exp_data_cfu.query(query_base)
    if len(tmp) == 0:
        query_base = f"strain1 == '{strain2}' and strain2 == '{strain1}'"
        tmp = exp_data_cfu.query(query_base)
        if len(tmp) == 0:
            print("something is wrong")

    query_0t = query_base + f" and timepoint == 0"
    if not CFU_per_sample:
        if strain_type == "CLIN":
            query_0t = query_0t + f" and pair == '{pair}'"
    else:
        if strain_type == "LAB" or strain_type == "MIX":
            query_0t = query_0t + f" and sample == {np.int64(sample)}"
        elif strain_type == "CLIN":
            query_0t = query_0t + f" and pair == '{pair}' and sample == {np.int64(sample)}"

    cfu_fractions = exp_data_cfu.query(query_0t)
    
    if cfu_fractions.empty:
        strain1 = key[1]
        strain2 = key[0]
        query_base = f"strain1 == '{strain1}' and strain2 == '{strain2}'"
        tmp = exp_data_cfu.query(query_base)
        if len(tmp) == 0:
            query_base = f"strain1 == '{strain2}' and strain2 == '{strain1}'"
            tmp = exp_data_cfu.query(query_base)
            if len(tmp) == 0:
                print("something is wrong")

        query_0t = query_base + f" and timepoint == {timepoint}"
        if not CFU_per_sample:
            if strain_type == "CLIN":
                query_0t = query_0t + f" and pair == '{pair}'"
        else:
            if strain_type == "LAB" or strain_type == "MIX":
                query_0t = query_0t + f" and sample == {np.int64(sample)}"
            elif strain_type == "CLIN":
                query_0t = query_0t + f" and pair == '{pair}' and sample == {np.int64(sample)}"

        cfu_fractions = exp_data_cfu.query(query_0t)
    
    # print("cfu fractions at " + str(timepoint) + ":")
    # display(cfu_fractions)
    return(cfu_fractions)
           
def read_config(input_config):
    with open(input_config) as user_file:
          parsed_json = json.load(user_file)
    return(parsed_json)

def get_input_config_values(parsed_json):
    input_node = parsed_json["input"]
    
    input_path = input_node["path"]
    TSB_simulations_separate = input_node["simulate TSB control experiments separately"]
    use_CFCM_parameters_as_start = input_node["use CFCM parameters as start points"] # only relevant for co-cultivation experiments 
    CFCM_parameter_paths = input_node["CFCM parameter paths"]                          # if use_CFCM_parameters_as_start == TRUE: give here the paths to the respective estimated parameters in the CFCM experiments
    fit_CFU_24h = input_node["fit CFU counts at 24h"]                                # bool, if CFU counts should be fitted at 24h timepoint ... this is only relevant for co-cultivation experiments
    CFU_counts_per_sample = input_node["CFU counts per sample"]                      # CFU counts at 0 and 24h will be used per sample (not mean over all samples); only relevant for co-cultivation experiments
    CFU_at_0timepoint = input_node["use CFU at 0h timepoint"]                        # CFU fractions at 0h will be used to set initial condition; only relevant for co-cultivation experiments
    CFU_0t_data_path = input_node["CFU 0h data path"]
    CFU_24t_data_path = input_node["CFU 24h data path"]
    
    print("input config values:")
    print("\t input path: \t " + input_path)
    print("\t simulate TSB control experiments separately: \t " + str(TSB_simulations_separate))
    print("\t use CFCM parameters as start points: " + str(use_CFCM_parameters_as_start))
    print("\t CFCM parameter paths: " + str(CFCM_parameter_paths))
    print("\t fit CFU counts at 24h: " + str(fit_CFU_24h))
    print("\t CFU counts per sample: " + str(CFU_counts_per_sample))
    print("\t use CFU at 0h timepoint: " + str(CFU_at_0timepoint))
    print("\t CFU 0h data path: " + CFU_0t_data_path)
    print("\t CFU 24h data path: " + CFU_24t_data_path)

    return (input_path, TSB_simulations_separate, use_CFCM_parameters_as_start, CFCM_parameter_paths, fit_CFU_24h, CFU_counts_per_sample, CFU_at_0timepoint, CFU_0t_data_path, CFU_24t_data_path)

def update_TSB_input_config_values(parsed_json):
    input_node = parsed_json["TSB"]["input"]
    
    use_CFCM_parameters_as_start = input_node["use CFCM parameters as start points"] # only relevant for co-cultivation experiments 
    fit_CFU_24h = input_node["fit CFU counts at 24h"]                                # bool, if CFU counts should be fitted at 24h timepoint ... this is only relevant for co-cultivation experiments
    CFU_at_0timepoint = input_node["use CFU at 0h timepoint"]                        # CFU fractions at 0h will be used to set initial condition; only relevant for co-cultivation experiments
        
    print("input config values:")
    print("\t use CFCM parameters as start points: " + str(use_CFCM_parameters_as_start))
    print("\t fit CFU counts at 24h: " + str(fit_CFU_24h))
    print("\t use CFU at 0h timepoint: " + str(CFU_at_0timepoint))

    return (use_CFCM_parameters_as_start, fit_CFU_24h, CFU_at_0timepoint)


def get_output_config_values(parsed_json):
    output_node = parsed_json["output"]
    
    output_path = output_node["path"]
    output_folder_add_time_stamp = output_node["output folder add time stamp"] 
    output_folder_extension = output_node["folder extension"] 
    save_raw_data_complete = output_node["save raw data complete"]                
    save_raw_data_N_best_points = output_node["save raw data number best points"] 
    
    print("output config values:")
    print("\t output path: " + output_path)
    print("\t add time stamp to output folder: " + str(output_folder_add_time_stamp))
    print("\t output folder extension: " + output_folder_extension)
    print("\t save raw data complete: " + str(save_raw_data_complete))
    print("\t save raw data N best points: " + str(save_raw_data_N_best_points))

    return (output_path, output_folder_add_time_stamp, output_folder_extension, save_raw_data_complete, save_raw_data_N_best_points)

def get_model_config_values(parsed_json):
    model_node = parsed_json["model"]
    
    strain_type = model_node["strain type"]      # different strain types: "LAB", "CLIN"
    exp_type = model_node["experiment type"]     # different experiments: "CFCM", "CoCultivation"
    condition = model_node["condition"]          # additional experimental condition: "Baseline", "Agr", "Mutations", "PhenotypicInteraction"
    model = model_node["model"]                  # Model: simple = normal logistic growth; complex = logistic growth with time variable carrying capacity
    species = model_node["species"]
    
    print("model config values:")
    print("\t strain type: " + strain_type)
    print("\t experiment type: " + exp_type)
    print("\t condition: " + condition)
    print("\t model: " + model)
    print("\t species: " + str(species))
    
    return (strain_type, exp_type, condition, model, species)

def update_TSB_model_config_values(parsed_json):
    model_node = parsed_json["TSB"]["model"]
    
    exp_type = model_node["experiment type"]     # different experiments: "CFCM", "CoCultivation"
    species = model_node["species"]
    
    print("model config values:")
    print("\t experiment type: " + exp_type)
    print("\t species: " + str(species))
    
    return (exp_type, species)

def get_fitting_config_values(parsed_json):
    global simulated_annealing_iterations

    fitting_node = parsed_json["fitting"]
    
    number_of_start_points = fitting_node["number of start points"]        
    simulated_annealing_iterations = fitting_node["simulated annealing iterations"] 
    get_only_min_solution = fitting_node["get only min solution"] 
    
    bounds = parsed_json["fitting"]["boundaries"]
    fitting_parameters = list(bounds.keys())
    
    print("fitting config values:")
    print("\t number of start points: " + str(number_of_start_points))
    print("\t get only min solution: " + str(get_only_min_solution))
    print("\t boundaries: " + str(bounds))
    
    return (number_of_start_points, get_only_min_solution, bounds, fitting_parameters)      

def get_CPU_config_values(parsed_json):
    cpu_node = parsed_json["cpu"]
    
    simulation_location = cpu_node["simulation location"]        
    number_cpu = cpu_node["number cpu"] 
    
    print("cpu config values:")
    print("\t simulation location: " + str(simulation_location))
    print("\t number cpu: " + str(number_cpu))
    
    return (simulation_location, number_cpu)  

def generate_output_foldername(model, strain_type, exp_type, condition, output_folder_add_time_stamp, output_folder_extension):
    time_stamp = ""
    if output_folder_add_time_stamp:
        now = datetime.now()
        time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S_")
    if output_folder_extension != "":
        output_folder_extension = "_" + output_folder_extension
    folder_name = time_stamp + strain_type + "_" + exp_type + "_" + condition + output_folder_extension
    print(folder_name)
    return folder_name

def tanh(t, p, g):
    return np.tanh(g*(t - p))

def C(t, C_start, C_end, p, g):
    return ((C_end + C_start)/2) + ((C_end - C_start)/2 * tanh(t, p, g))

@cfunc(lsoda_sig)
def ode_log_growth_complex(t, y, dy, params):
    y_ = nb.carray(y, (2,))
    params_ = nb.carray(params, (10,))
    for i in range(len(y_)):
        offset = int(i * (len(params_)/len(y_)))
        SP = y_[i]
        r_g = params_[offset]
        C_start = params_[offset + 1]
        C_end = params_[offset + 2]
        p = params_[offset + 3]
        g = params_[offset + 4]
        tanh = np.tanh(g * (t - p))
        C = ((C_end + C_start)/2) + (((C_end - C_start)/2) * tanh)
        if C == 0:
            dy[i] = np.inf
        else:
            dy[i] = r_g * SP * (1 - (SP/C))
    
### Analytic solution CFCM normal logistic growth
@njit
def analytic_normal_log_growth(time_steps, initials, params):
    # print("analytic_normal_log_growth ... ")
    num_species = len(initials)
    num_params = int(len(params)/num_species)
    solution = np.zeros((num_species, len(time_steps)), dtype="float64")
    for i in range(num_species):
        offset = i * num_params
        SP = initials[i]
        r_g = params[offset]
        C = params[offset + 1] 
        SP_sol = (SP * C)/(C * np.exp(-r_g * time_steps) + SP * (1-np.exp(-r_g*time_steps)))
        solution[i] = SP_sol
    return (solution)
    
funcptr_complex = ode_log_growth_complex.address 
@njit
def simulate(model, time_steps, initials, params):
    # print("simulate function is called...")
    num_species = len(initials)
    solution_fill_value = np.inf #-1000
    solution = np.zeros((num_species, len(time_steps)), dtype="float64")
    # print(params)
    if model == "simple":
        tmp_sol = analytic_normal_log_growth(time_steps, initials, params)
        out_sol = np.ones((num_species, tmp_sol.shape[1]))*solution_fill_value
        out_sol[:tmp_sol.shape[1]]  = tmp_sol
        solution = out_sol
    elif model == "complex":
        tmp_sol, _ = lsoda(funcptr_complex, initials, time_steps, data = params)
        out_sol = np.ones((num_species, tmp_sol.shape[0]))*solution_fill_value
        out_sol[:tmp_sol.T.shape[1]]  = tmp_sol.T
        solution = out_sol
    # print(solution)
    return solution

def objective(model, kinetics_data, cfu_data, time_steps, species, initials, start_point_index, df_list, **kwargs):
    # print("objective function is called...")
    # print(kwargs)
    fit_data = simulate(model, time_steps, initials, np.array(list(kwargs.values())))
    # display(fit_data)
    sum_fit_data = fit_data.sum(axis=0)
    kinetics_error = np.sum((sum_fit_data - kinetics_data)**2)
    
    # print("kinetics error = " + str(kinetics_error))
    cfu_error = 0
    if len(cfu_data) > 0:
        query = f"counted_specie == '{species[0]}'"
        exp_cfu_fraction_strain1 = float(cfu_data.query(query)['fraction'])
        # print("exp_cfu_fraction_strain1 = " + str(exp_cfu_fraction_strain1))
        if(fit_data[-1, 0] != np.inf and fit_data[-1, 1] != np.inf):
            # print(fit_data)
            value_strain1 = fit_data[0, -1]
            value_strain2 = fit_data[1, -1]
            # print(value_strain1)
            # print(value_strain2)
            sum_strains = value_strain1 + value_strain2
            # print(sum_strains)
            sim_cfu_fraction_strain1 = value_strain1/sum_strains*100
            # print("sim_cfu_fraction_strain1 = " + str(sim_cfu_fraction_strain1))
            cfu_error = np.square(exp_cfu_fraction_strain1 - sim_cfu_fraction_strain1)
            # print("cfu error = " + str(cfu_error))
        else:
            cfu_error = np.inf
    fun_eval = kinetics_error + cfu_error
    # print("fun_eval = " + str(fun_eval))
    df_list.append(pd.DataFrame(np.array([start_point_index,*kwargs.values(), fun_eval, kinetics_error, cfu_error]).reshape((1, len(kwargs.values()) + 4)),columns=["START_POINT",*kwargs.keys(), "EVAL", "EVAL_kinetics", "EVAL_CFU"]))
    return fun_eval

def loop_function(start_point_index, model, kinetics_data, cfu_data, time_steps, species, initials, starting_point, bounds, fitting_parameters): #, no_fitting_dict
    global simulated_annealing_iterations
    lw, up = bounds
    df_list = list()
    objective_wrapper = lambda params: objective(model, kinetics_data, cfu_data, time_steps, species, initials, start_point_index, df_list, **dict(zip(fitting_parameters, params)))  #, **dict(list(no_fitting_dict))
    ret = optimize.dual_annealing(objective_wrapper, bounds=list(zip(lw, up)), x0 = starting_point, maxiter = simulated_annealing_iterations, no_local_search = False)
    return pd.concat(df_list).reset_index(drop=True).reset_index().rename(columns={"index": "Step"})

def make_fitting(kinetics_data, cfu_data, time_steps, exp_type, model, species, bounds, starting_points, number_cpu, initials, fitting_parameters): 
    print("start make fitting")
    fitting_results = list()
    out = list()
    bounds_df = pd.DataFrame(bounds)
    lw = bounds_df.loc[0]
    up = bounds_df.loc[1]
    
    number_of_start_points = len(starting_points)
    input_multithreading = list(zip(np.arange(number_of_start_points), 
                                    [model] * number_of_start_points, 
                                    np.tile(kinetics_data,(number_of_start_points,1)),
                                    [cfu_data for i in range(number_of_start_points)] ,
                                    np.tile(time_steps,(number_of_start_points,1)), 
                                    np.tile(species,(number_of_start_points,1)), 
                                    np.tile(initials,(number_of_start_points,1)), 
                                    starting_points, 
                                    repeat((lw.values, up.values), number_of_start_points),
                                    np.tile(fitting_parameters, (number_of_start_points,1))))
    
    print("cores used: " + str(np.minimum(number_of_start_points, number_cpu)))
    with multiprocessing.Pool(np.minimum(number_of_start_points, number_cpu)) as pool:
        out = pool.starmap(loop_function, input_multithreading)
    print("fitting done")
    return (out)

def get_kinetics_best_fit(model, time_steps, initials, species, min_parameters):
    print("get kinetics for best parameters ... ")
    fitted_kinetics = pd.DataFrame()
    num_species = len(initials)
    sum_values = np.zeros(len(time_steps), dtype="float64")
    number_params = int(len(min_parameters)/num_species)
    values_species = pd.DataFrame()
    
    solution = np.zeros((num_species, len(time_steps)), dtype="float64")
    
    if model == "simple":
        tmp_sol = analytic_normal_log_growth(time_steps, initials, min_parameters)
        out_sol = np.ones((num_species, tmp_sol.shape[1])) * np.inf
        out_sol[:tmp_sol.shape[1]]  = tmp_sol
        sum_values = out_sol.sum(axis=0)
        if len(species) > 0:
            for i in range(len(initials)):
                values_current_specie_dict = dict({"time": time_steps, "value": out_sol.T[:,i], "data": "sim", "data_type": species[i]})
                values_current_specie_df = pd.DataFrame(values_current_specie_dict)
                values_species = pd.concat([values_species, values_current_specie_df])
        sum_data_dict = dict({"time": time_steps, "value": sum_values, "data": "sim", "data_type": "population"})
        sum_data_df = pd.DataFrame(sum_data_dict)
        if len(species) > 1:
            fitted_kinetics = pd.concat([sum_data_df, values_species])
        else:
            fitted_kinetics = pd.concat([sum_data_df])
    elif model == "complex":
        tmp_sol, _ = lsoda(funcptr_complex, initials, time_steps, data = min_parameters)
        out_sol = np.ones((num_species, tmp_sol.shape[0])) * np.inf
        out_sol[:tmp_sol.T.shape[1]]  = tmp_sol.T
        sum_values = out_sol.sum(axis=0)
        C_sim_df = pd.DataFrame()
        if len(species) > 0:
            for i in range(len(initials)):
                # print("i = " + str(i) + "; species = " + species[i])
                values_current_specie_dict = dict({"time": time_steps, "value": out_sol.T[:,i], "data": "sim", "data_type": species[i]})
                values_current_specie_df = pd.DataFrame(values_current_specie_dict)
                values_species = pd.concat([values_species, values_current_specie_df])
                
                specie_params = min_parameters[i*number_params:(i*number_params + (number_params))]
                C_paras_min = dict(zip(["C_start","C_end","p","g"], specie_params[1:5]))
                C_current_specie = C(time_steps, **C_paras_min)
                dtype_name = "C"
                if len(species) > 0:
                    dtype_name = "C_" + species[i]
                C_current_specie_dict = dict({"time": time_steps, "value": C_current_specie, "data": "sim", "data_type": dtype_name})
                C_current_specie_df = pd.DataFrame(C_current_specie_dict)
                C_sim_df = pd.concat([C_sim_df, C_current_specie_df])
        else:
            C_paras_min = dict(zip(["C_start","C_end","p","g"], min_parameters[1:5]))
            C_current_specie = C(time_steps, **C_paras_min)
            dtype_name = "C"
            if len(species) > 0:
                dtype_name = "C_" + species[i]
            C_current_specie_dict = dict({"time": time_steps, "value": C_current_specie, "data": "sim", "data_type": dtype_name})
            C_current_specie_df = pd.DataFrame(C_current_specie_dict)
            C_sim_df = pd.concat([C_sim_df, C_current_specie_df])
            
        sum_data_dict = dict({"time": time_steps, "value": sum_values, "data": "sim", "data_type": "population"})
        sum_data_df = pd.DataFrame(sum_data_dict)

        if len(species) > 1:
            fitted_kinetics = pd.concat([sum_data_df, values_species, C_sim_df])
        else:
            fitted_kinetics = pd.concat([sum_data_df, C_sim_df])
            
    return(fitted_kinetics)

def plot_kinetics(kinetics):
    fig = px.line(kinetics, x="time", y="value", color = "data_type", line_dash = "data_type")
    fig.update_layout(title={
                        'text': "In silico data",
                        'y':0.95,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                      xaxis_title="time [h]",
                      yaxis_title="value",
                      autosize=False, width=1000,  height=600,
                      font_size = 20, font_family="arial")
    fig.show()
    
def get_starting_points(key, strain_type, condition, model, species, number_of_start_points, bounds, use_CFCM_parameters_as_start, CFCM_parameter_paths):
    # print("get starting points function called ... ")
    start_points = pd.DataFrame()
    
    if len(species) >= 1:
            bounds_species = dict()
            for current_specie in species:
                curr_bounds = {k + '_' + current_specie: v for k, v in bounds.items()}
                bounds_species = bounds_species | curr_bounds
            bounds = bounds_species
    
    if use_CFCM_parameters_as_start:
        start_points = get_start_parameters_from_file(key, strain_type, condition, model, CFCM_parameter_paths)
    else:    
        sampler = qmc.LatinHypercube(d=len(bounds))
        sample = sampler.random(n = number_of_start_points)
        bounds_df = pd.DataFrame(bounds)
        lw = bounds_df.loc[0]
        up = bounds_df.loc[1]
        start_points = qmc.scale(sample, lw, up)
    return(start_points, bounds, list(bounds.keys()))

def get_start_parameters_from_file(key, strain_type, condition, model, CFCM_parameter_paths):
    print("get_start_parameters_from_file")
    strain1 = key[0]
    strain2 = key[1]
    specie1 = get_specie(strain1)
    specie2 = get_specie(strain2)
    pair = ""
    pair_int = 0
    
    if strain_type == "CLIN":
        pair = key[2]
        pair_int = int(pair)
        print("strain1 = " + strain1 + " (" + specie1 + ")" + ", strain2 = " + strain2   + " (" + specie2 + ")" + ", pair = " + pair)
    else: 
        print("strain1 = " + strain1 + " (" + specie1 + ")" + ", strain2 = " + strain2  + " (" + specie1 + ")")

    input_start_parameters_for_output = pd.DataFrame()
    start_parameters = pd.DataFrame()
    for CFCM_parameter_path in CFCM_parameter_paths:
        print("search for start parameters in: " + CFCM_parameter_path)
        CFCM_parameters = pd.read_csv(CFCM_parameter_path)
        # display(CFCM_parameters)
        
        if "supernatant" in CFCM_parameters.columns:
            query1 = f"strain == '{strain1}' and supernatant == '{strain2}'"
            if strain_type == "CLIN":
                query1 = query1 + f" and pair == {pair_int}"
            parameters1 = CFCM_parameters.query(query1)

            para_keys =["rg","C"]
            if model == "complex":
                para_keys = ["rg","C_start","C_end","p","g"]
            parameters1 = parameters1[para_keys]
            col_names = [key + "_" + specie1 for key in para_keys[:]]
            col_names_dict = dict(zip(para_keys[:], col_names))
            parameters1 = parameters1.rename(columns=col_names_dict)
            
            query2 = f"strain == '{strain2}' and supernatant == '{strain1}'"
            if strain_type == "CLIN":
                query2 = query2 + f" and pair == {pair_int}"
            parameters2 = CFCM_parameters.query(query2)
            parameters2 = parameters2[para_keys]
            col_names = [key + "_" + specie2 for key in para_keys[:]]
            col_names_dict = dict(zip(para_keys[:], col_names))
            parameters2 = parameters2.rename(columns=col_names_dict)
            
            parameters1 = parameters1.round(6)
            parameters1 = parameters1.drop_duplicates(ignore_index=True)
            parameters2 = parameters2.round(6)
            parameters2 = parameters2.drop_duplicates(ignore_index=True)
            parameters1["key"] = 1
            parameters2["key"] = 1
            if specie1 == "SA":
                start_parameters = pd.merge(parameters1, parameters2, on="key").drop("key", axis=1)
            else:
                start_parameters = pd.merge(parameters2, parameters1, on="key").drop("key", axis=1)

            start_parameters = start_parameters.reindex()
            start_parameters.index.name = 'index'

            input_start_parameters_for_output = pd.concat([input_start_parameters_for_output, start_parameters])

        else:
            currCols = ["rg_SA", "C_SA","rg_AB", "C_AB"]
            if model == "complex":
                currCols = ["rg_SA", "C_start_SA", "C_end_SA", "p_SA", "g_SA", 
                            "rg_AB", "C_start_AB", "C_end_AB", "p_AB", "g_AB"]
            
            success = False
            query = f"strain1 == '{strain1}' and strain2 == '{strain2}'"
            if strain_type == "CLIN":
                query = query + f" and pair == '{pair_int}'"
            currParameters = CFCM_parameters.query(query)
            currParameters = currParameters[currCols]
            if len(currParameters) > 0:
                success = True
            start_parameters = pd.concat([start_parameters, currParameters])

            query = f"strain1 == '{strain2}' and strain2 == '{strain1}'"
            if strain_type == "CLIN":
                query = query + f" and pair == '{pair_int}'"
            currParameters = CFCM_parameters.query(query)
            currParameters = currParameters[currCols]
            if len(currParameters) > 0:
                success = True
            start_parameters = pd.concat([start_parameters, currParameters])

            if not success:
                print("continue searching for start parameters")

                query = f"strain1 == '{strain1}' and strain2 != 'TSB'"
                if strain_type == "CLIN":
                    query = query + f" and pair == '{pair_int}'"
                currParameters = CFCM_parameters.query(query)
                currParameters = currParameters[currCols]
                start_parameters = pd.concat([start_parameters, currParameters])

                query = f"strain2 == '{strain1}'"
                if strain_type == "CLIN":
                    query = query + f" and pair == '{pair_int}'"
                currParameters = CFCM_parameters.query(query)
                currParameters = currParameters[currCols]
                start_parameters = pd.concat([start_parameters, currParameters])

                query = f"strain1 == '{strain2}' and strain2 != 'TSB'"
                if strain_type == "CLIN":
                    query = query + f" and pair == '{pair_int}'"
                currParameters = CFCM_parameters.query(query)
                currParameters = currParameters[currCols]
                start_parameters = pd.concat([start_parameters, currParameters])

                query = f"strain2 == '{strain2}'"
                if strain_type == "CLIN":
                    query = query + f" and pair == '{pair_int}'"
                currParameters = CFCM_parameters.query(query)
                currParameters = currParameters[currCols]
                start_parameters = pd.concat([start_parameters, currParameters])
                
            input_start_parameters_for_output = pd.concat([input_start_parameters_for_output, start_parameters])

    input_start_parameters_for_output = input_start_parameters_for_output.round(6)
    input_start_parameters_for_output = input_start_parameters_for_output.drop_duplicates(ignore_index=True)
    return(np.array(input_start_parameters_for_output))


def get_min_parameters(fitting_results, fitting_parameters):
    min_LSE = np.min(fitting_results.EVAL)
    print("min LSE = " + str(min_LSE))
    row = fitting_results.query('EVAL == @min_LSE')
    min_parameters = row[fitting_parameters]
    min_parameters = min_parameters.reset_index()
    min_parameters = min_parameters.drop_duplicates()
    min_parameters = min_parameters.drop(['index'], axis = 1)
    return(min_parameters, min_LSE)

def get_grouping(exp_data_kinetics, strain_type, exp_type, condition):
    gb = 0
    if exp_type == "CFCM":
        if strain_type == "LAB":
            if condition == "Baseline":
                gb = exp_data_kinetics.groupby(['strain','supernatant','sample','file_name'])
            elif condition == "Mutations":
                    gb = exp_data_kinetics.groupby(['strain','supernatant','mutation','sample','file_name'])
            elif condition == "Agr":
                    gb = exp_data_kinetics.groupby(['strain','supernatant','agr','sample','file_name'])
        elif strain_type == "CLIN":
            if condition == "Baseline":
                gb = exp_data_kinetics.groupby(['strain','supernatant', 'pair', 'sample', 'file_name'])
    elif exp_type == "CoCultivation":  
        if strain_type == "LAB" or strain_type == "MIX":
            if condition == "Baseline" or condition == "phenotypicInteraction":
                gb = exp_data_kinetics.groupby(['strain1','strain2','sample','file_name'])
        elif strain_type == "CLIN":
            if condition == "Baseline":
                gb = exp_data_kinetics.groupby(['strain1', 'strain2', 'pair', 'sample', 'file_name'])
    return(gb)

def get_specie(strain):
    if strain == "LS1" or strain == "USA300" or strain == "SA" or "SA" in strain:
        return "SA"
    elif strain == "A118" or strain == "A42" or strain == "AB" or "AB" in strain:
        return "AB"
    else: print("UNKOWN STRAIN: " + strain)

def get_sample_output_string(strain_type, exp_type, condition, key):
    print_str = ""
    file_name = ""
    if exp_type == "CFCM":
        if strain_type == "LAB":
            if condition == "Baseline":
                print_str = 'strain = ' + key[0] + '; supernatant = ' + key[1] + '; sample = ' + key[2]
                file_name = key[0] + "_Sup" + key[1] + "_sample-" + key[2]
            elif condition == "Mutations":
                print_str = 'strain = ' + key[0] + '; supernatant = ' + key[1] + '; mutation = ' + key[2] + '; sample = ' + key[3]
                file_name = key[0] + "_Sup" + key[1] + "_mutation-" + key[2] + "_sample-" + key[3]
            elif condition == "Agr":
                print_str = 'strain = ' + key[0] + '; supernatant = ' + key[1] + '; agr = ' + key[2] + '; sample = ' + key[3]
                file_name = key[0] + "_Sup" + key[1] + "_agr-" + key[2] + "_sample-" + key[3]
        elif strain_type == "CLIN":
            if condition == "Baseline":
                print_str = 'strain = ' + key[0] + '; supernatant = ' + key[1] + '; pair = ' + key[2] + '; sample = ' + key[3]
                file_name = key[0] + "_Sup" + key[1] + "_pair-" + key[2] + "_sample-" + key[3]
    elif exp_type == "CoCultivation":
        if strain_type == "LAB" or strain_type == "MIX":
            if condition == "Baseline" or condition == "phenotypicInteraction":
                print_str = 'strain1 = ' + key[0] + '; strain2 = ' + key[1] + '; sample = ' + key[2]
                file_name = key[0] + "_" + key[1] + "_sample-" + key[2]
        elif strain_type == "CLIN" :
            if condition == "Baseline":
                print_str = 'strain1 = ' + key[0] + '; strain2 = ' + key[1] + '; pair = ' + key[2] + '; sample = ' + key[3]
                file_name = key[0] + "_" + key[1] + "_pair-" + key[2] + "_sample-" + key[3]
    return(print_str, file_name)

def add_meta_data_to_df(df, key, strain_type, exp_type, condition):
    if exp_type == "CFCM":
        df['strain'] = key[0]
        df['supernatant'] = key[1]
        if strain_type == "LAB":
            if condition == "Baseline":
                df = df.assign(sample = key[2], file_name = key[3])
            elif condition == "Mutations":
                df = df.assign(mutation = key[2], sample = key[3], file_name = key[4])
            elif condition == "Agr":
                df = df.assign(agr = key[2], sample = key[3], file_name = key[4])
        elif strain_type == "CLIN":
            if condition == "Baseline":
                df = df.assign(pair = key[2], sample = key[3], file_name = key[4])
    elif exp_type == "CoCultivation":
        df['strain1'] = key[0]
        df['strain2'] = key[1]
        if strain_type == "LAB" or strain_type == "MIX":
            df = df.assign(sample = key[2], file_name = key[3])
        elif strain_type == "CLIN":
            df = df.assign(pair = key[2], sample = key[3], file_name = key[4])
    return(df)
            
