import re
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from feature_extraction import FeatureExtraction
import constants
from copy import deepcopy
import constants
import os


def time_to_seconds(t):
  """
    Converts a time (string) with a format of hh:mm:ss to seconds

    Parameters:
        t (str): time with a format of hh:mm:ss

    Returns:
        int: number of seconds in t
  """

  m = re.match("^(\d{1,2}):(\d{1,2}):(\d{1,2})$", t)

  if not m:
    return None

  return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))


def plot_seizures(durations):
    """
      Plots timelines including seizure events as red areas

      Parameters:
          durations (list): output of chbmit_file_reader.extract_files_info_from_summary function to be plotted
    """

    fig, axs = plt.subplots(len(durations), 1, figsize=(20, len(durations)))

    for duration_idx, duration in enumerate(durations):
        axs[duration_idx].set_ylim(0, 1)
        axs[duration_idx].set_xlim(duration_idx*3600, (duration_idx+1)*3600)
        axs[duration_idx].set_yticks([])

        start_arr, end_arr = [], []
        for start, end in list(zip(*[3600*duration_idx + np.array(duration[s]) for s in ['seizure_start_times', 'seizure_end_times']])):
            axs[duration_idx].fill_between((start, end), 0, 1, facecolor='red', alpha=0.2)
            start_arr.append(start)
            end_arr.append(end)

        axs[duration_idx].set_xticks(start_arr)
        axs[duration_idx].secondary_xaxis("top").set_xticks(end_arr)
        
    fig.tight_layout()


def get_non_seizure_windows_adjacent_to_seizure_windows_mask(labels, expansion):
    """
      Outputs a mask showing non-seizure windows adjacent to seizure windows

      Parameters:
          labels (np.ndarray): Array of labels (0 or 1 indicating seizure or non-seizure events)
          expansion (int): Maximum distance of non-seizure windows to be fetched from seizure windows (in their neighborhood)
    """

    res = labels.copy()
    for exp in range(1, expansion+1):
        res = res | np.roll(labels, exp) | np.roll(labels, -exp)
    res = res & (~labels)

    return res.astype(bool)


def sample_from_windows(
    windows,
    labels,
    zero_to_one_ration,
    retain_non_seizure_windows_adjacent_to_seizure_windows_expansion=0,
    retain_non_seizure_windows_adjacent_to_seizure_windows_prob_ratio=5
  ):
  """
    Subsamples windows (used for (offline) training set) 

    Parameters:
        windows (list): List of windows
        labels (list): List of labels
        zero_to_one_ration (int): ratio indicating the number of seizure windows to non-seizure windows
        retain_non_seizure_windows_adjacent_to_seizure_windows_expansion (int): Parameter used for controlling the action of sampling more 
                                                                                instances from the neighborhood of seizure windows (Maximum 
                                                                                distance of non-seizure windows to be fetched from seizure windows)
        retain_non_seizure_windows_adjacent_to_seizure_windows_prob_ratio (int): Parameter used for controlling the action of sampling more 
                                                                                 instances from the neighborhood of seizure windows (Probability ratio
                                                                                 of non-seizure windows near the seizures to other non-seizure
                                                                                 windows in the sampling process)

    Returns:
        list: Subsampled windows
        list: Subsampled labels
  """

  assert type(labels) == np.ndarray
  
  num_of_zeros, num_of_ones = [np.sum(labels == i) for i in range(2)]
  assert num_of_zeros - num_of_ones*zero_to_one_ration >= 0

  windows, labels = deepcopy(windows), deepcopy(labels)
  
  num_to_remove_from_zeros = num_of_zeros - num_of_ones*zero_to_one_ration

  remove_options = np.where(labels == 0)[0]

  p = np.ones(labels.shape[0])
  expanded_labels = get_non_seizure_windows_adjacent_to_seizure_windows_mask(labels, retain_non_seizure_windows_adjacent_to_seizure_windows_expansion)
  p[expanded_labels] = 1/retain_non_seizure_windows_adjacent_to_seizure_windows_prob_ratio
  p = p[labels == 0]
  p = p / p.sum()

  remove_indices = np.sort(np.random.choice(remove_options, num_to_remove_from_zeros, replace=False, p=p))[::-1]

  for idx in remove_indices:
    windows.pop(idx)
  labels = np.delete(labels, remove_indices)

  return windows, labels


def extract_features_and_convert_to_tensor(windows, labels):
  """
    Extracts features from windows using FeatureExtraction 

    Parameters:
        windows (list): List of windows
        labels (list): List of labels

    Returns:
        list: New windows after feature extraction phase
        list: New labels after feature extraction phase
  """

  labels = torch.from_numpy(labels).type(torch.LongTensor)
  windows = [torch.from_numpy(w) for w in windows]

  for idx, w in enumerate(tqdm(windows)):
    windows[idx] = torch.from_numpy(np.concatenate([FeatureExtraction(w[i], constants.fs).get_features() for i in range(w.shape[0])]))

  windows = torch.stack(windows, dim=0)

  return windows, labels


def plot_loss_and_accuracy(train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr):
  """
    Plots loss and accuracy at each epoch

    Parameters:
        train_loss_arr (list): List of training dataset loss at each epoch
        val_loss_arr (list): List of validation dataset loss at each epoch
        train_acc_arr (list): List of training dataset accuracy at each epoch
        val_acc_arr (list): List of validation dataset accuracy at each epoch
  """

  fig, axs = plt.subplots(1, 2, figsize=(21, 7))

  axs[0].set_title("Loss plot")
  axs[0].plot(train_loss_arr)
  axs[0].plot(val_loss_arr)
  axs[0].legend(["Train Loss", "Validation Loss"]);

  axs[1].set_title("Accuracy plot")
  axs[1].plot(train_acc_arr)
  axs[1].plot(val_acc_arr)
  axs[1].legend(["Train Accuracy", "Validation Accuracy"]);

  plt.savefig(os.path.join(constants.RESULT_PATH, 'train.png'), facecolor='white', bbox_inches='tight')


def get_accuracy_and_sensitivity_over_time(model, test_dataset):
  """
    Calculates accuracy and sensitivity over time of a given model on a specific dataset

    Parameters:
        model (torch.nn.Module): Model
        test_dataset (torch.utils.data.Dataset): Dataset
    
    Returns:
        list: Array of accuracy over time
        list: Array of sensitivity over time
  """

  acc_arr_without_update = []
  sen_arr_without_update = []

  model.eval()
  cnt_correct, cnt_total = 0, 0
  cnt_correct_seizure = 0
  for x, y in tqdm(test_dataset):
    outputs = model(x.view(1, -1).type(torch.FloatTensor)).detach()
    pred = torch.max(outputs, axis=1)[1].item()
    
    cnt_total += 1
    if pred == y:
      cnt_correct += 1
      if y == 1:
        cnt_correct_seizure += 1
    
    acc_arr_without_update.append(cnt_correct/cnt_total)
    sen_arr_without_update.append(cnt_correct_seizure/test_dataset[:][1].sum())
  
  return acc_arr_without_update, sen_arr_without_update


def draw_accuracy_and_sensitivity_over_time(acc_arr_without_update, sen_arr_without_update, test_dataset, acc_arr_with_update=None, sen_arr_with_update=None):
  """
    Plots accuracy and sensitivity over time

    Parameters:
        acc_arr_without_update (list): Array of accuracy over time for a model that was trained just offline
        sen_arr_without_update (list): Array of sensitivity over time for a model that was trained just offline
        test_dataset (torch.utils.data.Dataset): Dataset
        acc_arr_with_update (list|None): Array of accuracy over time for a model that was trained offline and online
        sen_arr_with_update (list|None): Array of sensitivity over time for a model that was trained offline and online
  """

  _, axs = plt.subplots(1, 2, figsize=(15, 5))
  for idx, (arr_without_update, arr_with_update, title) in enumerate(zip(
      [acc_arr_without_update, sen_arr_without_update], 
      [acc_arr_with_update, sen_arr_with_update], 
      ["Accuracy", "Sensitivity"] 
    )):
    axs[idx].plot(arr_without_update)
    if arr_with_update is not None:
      axs[idx].plot(arr_with_update)
      axs[idx].legend(['Offline Training Only', 'Adaptive Learning']);
    axs[idx].fill_between(range(0, test_dataset.labels.shape[0]), 0, 1, where=test_dataset.labels, facecolor='red', alpha=0.1)
    axs[idx].set_ylim(0, 1.005);
    axs[idx].set_title(f"{title} over time")


def get_raw_predictions(model, dataset):
  """
    Calculates raw predictions of a model for a given dataset

    Parameters:
        model (torch.nn.Module): Model
        test_dataset (torch.utils.data.Dataset): Dataset
    
    Returns:
        list: Tensor of raw predictions with a shape of (len(dataset), 2)
  """

  raw_predictions = []

  model.eval()
  for x, y in tqdm(dataset):
    output = model(x.view(1, -1).type(torch.FloatTensor)).detach().squeeze()
    raw_predictions.append(output)
    
  return torch.stack(raw_predictions)
