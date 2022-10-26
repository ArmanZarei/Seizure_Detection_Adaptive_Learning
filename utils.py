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
  """Converts a time (string) with format of hh:mm:ss to seconds"""
  m = re.match("^(\d{1,2}):(\d{1,2}):(\d{1,2})$", t)

  if not m:
    return None

  return int(m.group(1))*3600 + int(m.group(2))*60 + int(m.group(3))


def plot_seizures(durations):
    fig, axs = plt.subplots(len(durations), 1, figsize=(20, len(durations)))

    for duration_idx, duration in enumerate(durations):
        axs[duration_idx].set_ylim(0, 1)
        axs[duration_idx].set_xlim(duration_idx*3600, (duration_idx+1)*3600)
        axs[duration_idx].set_yticks([])

        # arr = [duration_idx*3600]
        start_arr, end_arr = [], []
        for start, end in list(zip(*[3600*duration_idx + np.array(duration[s]) for s in ['seizure_start_times', 'seizure_end_times']])):
            axs[duration_idx].fill_between((start, end), 0, 1, facecolor='red', alpha=0.2)
            start_arr.append(start)
            end_arr.append(end)

        axs[duration_idx].set_xticks(start_arr)
        axs[duration_idx].secondary_xaxis("top").set_xticks(end_arr)
        
    fig.tight_layout()


def get_non_seizure_windows_adjacent_to_seizure_windows_mask(labels, expansion):
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
  labels = torch.from_numpy(labels).type(torch.LongTensor)
  windows = [torch.from_numpy(w) for w in windows]

  for idx, w in enumerate(tqdm(windows)):
    windows[idx] = torch.from_numpy(np.concatenate([FeatureExtraction(w[i], constants.fs).get_features() for i in range(w.shape[0])]))

  windows = torch.stack(windows, dim=0)

  return windows, labels


def plot_loss_and_accuracy(train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr):
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
