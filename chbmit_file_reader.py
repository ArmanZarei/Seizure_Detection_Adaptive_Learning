import re
from pathlib import Path
import os
from utils import time_to_seconds
from tqdm import tqdm
import constants
from IPython.display import clear_output
import mne
import numpy as np
import wget


def extract_files_info_from_summary(path_to_file):
  """
    Extracts information from the summary of a specific patient

    Parameters:
        path_to_file (str): Path to the summary file

    Returns:
        list: List of files related to a specific patient along side some additional information
              Example: 
                  [
                    {
                      'seizure_start_times': [2670],
                      'seizure_end_times': [2841],
                      'file_name': 'chb08_02.edf',
                      'start_time': 44937,
                      'end_time': 48537,
                      'num_of_seizures': 1
                    },
                    ...
                  ]
  """

  with open(path_to_file, "r") as f:
    lines = f.readlines()[29:]

  durations_lines_group = list(map(lambda x: x.split("\n"), "".join(lines).split("\n\n")))
  durations = []
  for durations_lines in durations_lines_group:
    if len(durations_lines) == 1 and durations_lines[0] == '':
      continue
    
    duration = {
      'seizure_start_times': [],
      'seizure_end_times': []
    }

    for duration_line in durations_lines:
      m = re.match("^File Name: (.*\.edf)$", duration_line)
      if m:
        duration['file_name'] = m.group(1)
      
      m = re.match("^File Start Time: (.*)$", duration_line)
      if m:
        duration['start_time'] = time_to_seconds(m.group(1))
      
      m = re.match("^File End Time: (.*)$", duration_line)
      if m:
        duration['end_time'] = time_to_seconds(m.group(1))
      
      m = re.match("^Number of Seizures in File: (\d+)$", duration_line)
      if m:
        duration['num_of_seizures'] = int(m.group(1))
      
      m = re.match("^Seizure( \d*\s*)Start Time: (\d+) seconds$", duration_line)
      if m:
        duration['seizure_start_times'].append(int(m.group(2)))
      
      m = re.match("^Seizure( \d*\s*)End Time: (\d+) seconds$", duration_line)
      if m:
        duration['seizure_end_times'].append(int(m.group(2)))
      
    if len(duration['seizure_start_times']) != len(duration['seizure_end_times']) or len(duration['seizure_start_times']) != duration['num_of_seizures']:
      raise Exception("Seizure start times array length differs from seizure end times array length or from number of seizures stated in the summary file")

    durations.append(duration)
  
  return durations


def download_patient_summary_and_extract_files_information(dest_dir, patient):
  """
    Downloads a patient's summary file and data files and extracts some information from them

    Parameters:
        dest_dir (str): The destination directory that other files would be downloaded to
        patient (str): Name of the patient (Example: "chb08")
    
    Returns:
        list: List of files related to a specific patient along side some additional information
              Example: 
                  [
                    {
                      'seizure_start_times': [2670],
                      'seizure_end_times': [2841],
                      'file_name': 'chb08_02.edf',
                      'start_time': 44937,
                      'end_time': 48537,
                      'num_of_seizures': 1
                    },
                    ...
                  ]
  """

  if not os.path.isdir(dest_dir):
    Path(dest_dir).mkdir(parents=True) # Create destination directory if not exists

  patient_url = f"{constants.ROOT_URL}/{patient}" # URL of the patient's dataset

  patient_summary_file_path = f"{os.path.join(dest_dir, patient)}-summary.txt"
  if not os.path.exists(patient_summary_file_path):
    wget.download(
      url=f"{patient_url}/{patient}-summary.txt?download", 
      out=patient_summary_file_path
    )
  
  durations = extract_files_info_from_summary(patient_summary_file_path) # Extract information about files from summary file

  for duration in tqdm(durations):
    patient_file_path = f"{dest_dir}/{duration['file_name']}"
    if not os.path.exists(patient_file_path):
      wget.download(
        url=f"{patient_url}/{duration['file_name']}?download",
        out=patient_file_path
      )
  
  return durations


def extract_windows_and_labels_of_patient(patient_data_dir, durations, duration_freq):
  """
    Extracts windows and labels from files of a patient

    Parameters:
        patient_data_dir (str): Directory of the patient's data
        durations (list): List of files related to a specific patient along side some additional information (Extracted from above functions)
        duration_freq (int): Frequency that is used for setting windows sizes
    
    Returns:
        list: Windows extracted from patient's files
        list: Labels extracted from patient's files
  """
  
  windows = []
  labels = []

  for duration in tqdm(durations):
    raw = mne.io.read_raw_edf(f'{patient_data_dir}/{duration["file_name"]}') # Read each file

    clear_output(wait=True) # Clear the output

    data, times = raw[:] # Get data/times of each file

    windows = windows + np.split(data, data.shape[1]//duration_freq, axis=1) # Split data based on duration_freq

    if duration['num_of_seizures'] > 0:
      is_time_inside_seizure_duration = (times > duration["seizure_start_times"][0]) & (times < duration["seizure_end_times"][0]) # Is window inside a seizure window
      for seizure_event_idx in range(1, duration['num_of_seizures']):
        is_time_inside_seizure_duration = is_time_inside_seizure_duration | ((times > duration["seizure_start_times"][seizure_event_idx]) & (times < duration["seizure_end_times"][seizure_event_idx]))
      labels = labels + list(map(
        lambda x: 1 if np.all(x) else 0 if np.all(~x) else -1,\
        np.split(is_time_inside_seizure_duration, data.shape[1]//duration_freq)
      )) # Label 1/0 for seizure/non-sizure and -1 for a window overlapping both
    else:
      labels = labels + np.zeros(data.shape[1]//duration_freq, dtype=int).tolist()

  # Remove windows that have an overlap with both non-seizure and seizure durations
  for i in range(len(labels)-1, -1, -1):
    if labels[i] == -1:
      windows.pop(i)
      labels.pop(i)
  
  labels = np.array(labels)

  return windows, labels
