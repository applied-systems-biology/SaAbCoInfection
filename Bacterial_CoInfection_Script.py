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

from matplotlib.colors import LinearSegmentedColormap as LSCol
import numpy as np
import itertools
import scipy.optimize as fit
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm_notebook
import time  
import os
import shutil
import sys
from IPython.display import display
from scipy.integrate import solve_ivp

import coInfectionModel_functions as coinf
# %load_ext autoreload
# %autoreload 2
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# input_config = "./json_input_configs/Baseline_LAB_CFCM_srv.json"
print('cmd entry:', sys.argv)
input_config = sys.argv[1]
print("input config: " + input_config)
input_base_path, config_filename = os.path.split(input_config)
parsed_json = coinf.read_config(input_config)

### parameters from config ###
input_path, TSB_simulations_separate, use_CFCM_parameters_as_start, CFCM_parameter_paths, fit_CFU_24h, CFU_counts_per_sample, CFU_at_0timepoint, CFU_0t_data_path, CFU_24t_data_path =  coinf.get_input_config_values(parsed_json)
output_path, output_folder_add_time_stamp, output_folder_extension, save_raw_data_complete, save_raw_data_N_best_points = coinf.get_output_config_values(parsed_json)
strain_type, exp_type, condition, model, species = coinf.get_model_config_values(parsed_json)
number_of_start_points, get_only_min_solution, bounds, fitting_parameters = coinf.get_fitting_config_values(parsed_json)
simulation_location, number_cpu = coinf.get_CPU_config_values(parsed_json)

exp_data_path = ""
if simulation_location == "local":
    exp_data_path = os.path.join(input_path, "GrowthCurves", strain_type, exp_type, condition, "model_input/Tanh_Model/srv/experiment")
else:
    exp_data_path = os.path.join(input_path, "input", strain_type, exp_type, condition, "experiment")
print(exp_data_path)
exp_data_kinetics = coinf.get_experimental_data(exp_data_path, strain_type, exp_type, condition)

# query = "strain1 == 'A42' and strain2 == 'LS1' and sample == '1'" ### just for debugging!
# exp_data_kinetics = exp_data_kinetics.query(query)                ### just for debugging!

exp_data_kinetics_TSB = pd.DataFrame()
if TSB_simulations_separate:
    query = "strain2 == 'TSB'"
    exp_data_kinetics_TSB = exp_data_kinetics.query(query)
    exp_data_kinetics_TSB = exp_data_kinetics_TSB.reset_index()
    exp_data_kinetics_TSB.rename(columns={'strain1':'strain', 'strain2': 'supernatant'}, inplace=True)
    query = "strain2 != 'TSB'"
    exp_data_kinetics = exp_data_kinetics.query(query)

# exp_data_cfu = coinf.get_exp_data_CFU_counts(strain_type, exp_type, CFU_at_0timepoint, CFU_0t_data_path, CFU_24t_data_path)

output_folder = coinf.generate_output_foldername(model, strain_type, exp_type, condition, output_folder_add_time_stamp, output_folder_extension)
save_path = ""
exp_data_path = ""
if simulation_location == "local":
    save_path = os.path.join(output_path, "GrowthCurves", strain_type, exp_type, condition, output_folder)
else:
    save_path = os.path.join(output_path, strain_type, exp_type, condition, output_folder)
print("path to experimental data: " + exp_data_path)

if os.path.exists(save_path) == False:
    os.makedirs(save_path)
print("path where results will be saved: " + save_path)
shutil.copyfile(input_config, os.path.join(save_path, config_filename))

gb = coinf.get_grouping(exp_data_kinetics, strain_type, exp_type, condition)
all_fitting_parameters = pd.DataFrame()
all_fitted_kinetics = pd.DataFrame()

i = 1
n_samples = str(len(gb))
for key, curr in gb:
    print(str(i) + " / " + n_samples)
    i += 1
    print_str, file_name = coinf.get_sample_output_string(strain_type, exp_type, condition, key)
    print(print_str)

    initials = np.array(curr["value"].iloc[0])
    if exp_type == "CoCultivation":
        if CFU_at_0timepoint:
            cfu_fractions_0t = coinf.get_CFU_fractions(key, species, strain_type, exp_type, CFU_0t_data_path, CFU_counts_per_sample, 0)
            factors = np.array([float(cfu_fractions_0t.query(f"counted_specie == '{species[0]}'")['fraction'])/100, float(cfu_fractions_0t.query(f"counted_specie == '{species[1]}'")['fraction'])/100])
            initials = curr["value"].iloc[0] * factors
        else:
            initials = np.array([curr["value"].iloc[0]/2, curr["value"].iloc[0]/2])
    time_steps = curr["time"]
    exp_data = curr["value"]
    
    cfu_fractions_24t = []
    if fit_CFU_24h and exp_type == "CoCultivation":
        cfu_fractions_24t = coinf.get_CFU_fractions(key, species, strain_type, exp_type, CFU_24t_data_path, CFU_counts_per_sample, 24)
    
    starting_points, bounds_new, fitting_parameters_new = coinf.get_starting_points(key, strain_type, condition, model, species, number_of_start_points, bounds, use_CFCM_parameters_as_start, CFCM_parameter_paths)
    print("starting points: (" + str(len(starting_points)) + ")" )

    start_time = time.time()
    fitting_result = coinf.make_fitting(exp_data, cfu_fractions_24t, time_steps, exp_type, model, species, bounds_new, starting_points, number_cpu, initials, fitting_parameters_new)
    end_time = time.time()
    diff_time = end_time - start_time
    print("Elapsed time: " + time.strftime("%H:%M:%S", time.gmtime(diff_time)))
    
    fitting_result = pd.concat(fitting_result)
    curr_path = os.path.join(save_path, "raw_data", file_name)
    if os.path.exists(curr_path) == False:
        os.makedirs(curr_path)
        
    if save_raw_data_complete:
        fitting_result.to_hdf((curr_path + "/fitting_raw_data.h5"), key='stage', mode='w')
    else:
        fitting_result_subset = (fitting_result.groupby("START_POINT", group_keys=False)
                                 .apply(lambda grp_:grp_
                                        .sort_values("EVAL",ascending=False).iloc[-save_raw_data_N_best_points:]
                                        .reset_index(drop=True)).reset_index(drop=True))
        fitting_result_subset.to_hdf((curr_path + "/fitting_raw_data.h5"), key='stage', mode='w')

    min_parameters, EVAL = coinf.get_min_parameters(fitting_result, fitting_parameters_new)
    if len(species) == 0:
        initials = np.array([initials])
    else:
        initials = np.array(initials)
    fitted_kinetics = coinf.get_kinetics_best_fit(model, np.array(time_steps), initials, species, np.array(min_parameters.values.tolist()[0]))
    fitted_kinetics = coinf.add_meta_data_to_df(fitted_kinetics, key, strain_type, exp_type, condition)
    min_parameters = coinf.add_meta_data_to_df(min_parameters, key, strain_type, exp_type, condition)

    min_parameters["EVAL"] = EVAL
    all_fitted_kinetics = pd.concat([all_fitted_kinetics, fitted_kinetics], ignore_index=True)
    all_fitting_parameters = pd.concat((all_fitting_parameters, min_parameters))
    
all_kinetics = pd.concat([exp_data_kinetics, all_fitted_kinetics], ignore_index=True)
all_kinetics.to_csv((save_path + "/fitted_kinetics.csv"))
all_fitting_parameters.to_csv((save_path + "/min_parameters.csv"))


if TSB_simulations_separate:
    
    all_fitting_parameters = pd.DataFrame()
    all_fitted_kinetics = pd.DataFrame()
    all_fitted_kinetics = pd.DataFrame()
    
    exp_type, species = coinf.update_TSB_model_config_values(parsed_json)
    use_CFCM_parameters_as_start, fit_CFU_24h, CFU_at_0timepoint = coinf.update_TSB_input_config_values(parsed_json)

    gb = coinf.get_grouping(exp_data_kinetics_TSB, strain_type, exp_type, condition)

    p_bar = tqdm_notebook(gb)
    for key, curr in p_bar:
        print_str, file_name = coinf.get_sample_output_string(strain_type, exp_type, condition, key)
        print("filename = " + file_name)
        print(print_str)

        initials = np.array(curr["value"].iloc[0])
        time_steps = curr["time"]
        exp_data = curr["value"]

        cfu_fractions_24t = []
        starting_points, bounds_new, fitting_parameters_new = coinf.get_starting_points(key, strain_type, condition, model, species, number_of_start_points, bounds, use_CFCM_parameters_as_start, CFCM_parameter_paths)
        print("starting points: (" + str(len(starting_points)) + ")" )

        start_time = time.time()
        fitting_result = coinf.make_fitting(exp_data, cfu_fractions_24t, time_steps, exp_type, model, species, bounds_new, starting_points, number_cpu, initials, fitting_parameters_new)
        end_time = time.time()
        diff_time = end_time - start_time
        print("Elapsed time: " + time.strftime("%H:%M:%S", time.gmtime(diff_time)))

        fitting_result = pd.concat(fitting_result)
        curr_path = os.path.join(save_path, "raw_data", file_name)
        if os.path.exists(curr_path) == False:
            os.makedirs(curr_path)

        if save_raw_data_complete:
            fitting_result.to_hdf((curr_path + "/fitting_raw_data.h5"), key='stage', mode='w')
        else:
            fitting_result_subset = (fitting_result.groupby("START_POINT", group_keys=False)
                                     .apply(lambda grp_:grp_
                                            .sort_values("EVAL",ascending=False).iloc[-save_raw_data_N_best_points:]
                                            .reset_index(drop=True)).reset_index(drop=True))
            fitting_result_subset.to_hdf((curr_path + "/fitting_raw_data.h5"), key='stage', mode='w')
        
        min_parameters, EVAL = coinf.get_min_parameters(fitting_result, fitting_parameters_new)
        if len(species) == 0:
            initials = np.array([initials])
        else:
            initials = np.array(initials)
        fitted_kinetics = coinf.get_kinetics_best_fit(model, np.array(time_steps), initials, species, np.array(min_parameters.values.tolist()[0]))
        fitted_kinetics = coinf.add_meta_data_to_df(fitted_kinetics, key, strain_type, exp_type, condition)
        min_parameters = coinf.add_meta_data_to_df(min_parameters, key, strain_type, exp_type, condition)

        min_parameters["EVAL"] = EVAL
        all_fitted_kinetics = pd.concat([all_fitted_kinetics, fitted_kinetics], ignore_index=True)
        all_fitting_parameters = pd.concat((all_fitting_parameters, min_parameters))

    all_kinetics = pd.concat([exp_data_kinetics_TSB, all_fitted_kinetics], ignore_index=True)
    all_kinetics.to_csv((save_path + "/fitted_kinetics_TSB.csv"))
    all_fitting_parameters.to_csv((save_path + "/min_parameters_TSB.csv"))

print("Fitting for all samples done! Results can be found here: " + save_path)