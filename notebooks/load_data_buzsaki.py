
# Import matlab file from Buzsaki lab and save to spike time dictionary (pickle)
import os
import numpy as np
import scipy
import pickle

def spike_time_dict_from_mat_buzsaki(mat_file):
    mat_data = scipy.io.loadmat(mat_file)
    dtypes = mat_data["spikes"].dtype.names
    # print(dtypes)
    i_UID = np.where(np.array(dtypes) == "UID")[0][0]
    i_times = np.where(np.array(dtypes) == "times")[0][0]
    IDs = mat_data["spikes"][0][0][i_UID][0]
    spike_times = mat_data["spikes"][0][0][i_times][0]
    n_IDs = len(IDs)
    # Spike time dictionary
    spike_time_dict = {}
    for i in range(n_IDs):
        spike_time_dict[IDs[i]] = spike_times[i][:, 0]
    return spike_time_dict

# Paths
project_path = "/Users/ilgaz/Desktop/data_share_20260429"
input_data_path = os.path.join(project_path, "data/buzsaki")
output_data_path = os.path.join(project_path, "data/spike_time_dicts")

# Input and output files.

###############################################################
#### Specify input file name here.


# # Specified.
# input_file_names = ["BWRat18_020513.spikes.cellinfo.mat"]

# All files in input data path.
input_file_names = [f for f in os.listdir(input_data_path) if f.endswith(".spikes.cellinfo.mat")]

###############################################################


# Load data 
for input_file_name in input_file_names:
    mat_file = os.path.join(input_data_path, input_file_name)
    spike_time_dict = spike_time_dict_from_mat_buzsaki(mat_file)
    # Save with pickle
    input_animal_session = input_file_name.split(".")[0]
    output_file_name = "buzsaki_%s.pkl" % input_animal_session
    output_file = os.path.join(output_data_path, output_file_name)
    with open(output_file, "wb") as f:
        pickle.dump(spike_time_dict, f)

