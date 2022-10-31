import sklearn.metrics 
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import seaborn as sns
import numpy as np
import constants
import os


def evaluate_model(model, data_loader, print_output=True, return_output=False):
  """
    Evaluates a model by calculating confusion matrix, sensitivity, specificity, auc and roc

    model (torch.nn.Module): Model
    data_loader (torch.utils.data.DataLoader): Dataloader of test dataset
    print_output (bool): A boolean indicating whether the output should be printed or not
    return_output (bool): A boolean indicating whether the output should be returned or not
  """

  model.eval()
  y_true = data_loader.dataset[:][1]
  y_pred = torch.max(model(data_loader.dataset[:][0].type(torch.FloatTensor)), axis=1)[1]
  pred_prob = F.softmax(model(data_loader.dataset[:][0].type(torch.FloatTensor)), dim=1)[:, 1].detach()

  confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)

  if print_output:
    plt.figure(figsize=(7, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap='Blues')
    plt.ylabel('Ground Truth', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.savefig(os.path.join(constants.RESULT_PATH, 'mat.png'), facecolor='white', bbox_inches='tight')
    plt.show()

  sensitivity = confusion_matrix[1, 1]/np.sum(confusion_matrix, axis=1)[1]
  if print_output:
    print('\nSensitivity : ', sensitivity)

  specificity = confusion_matrix[0, 0]/np.sum(confusion_matrix, axis=1)[0]
  if print_output:
    print('Specificity : ', specificity)

  f1_score = sklearn.metrics.f1_score(y_true, y_pred)
  if print_output:
    print("F1-Score:", f1_score)

  fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, pred_prob)
  auc = sklearn.metrics.auc(fpr, tpr)
  if print_output:
    print("AUC:", auc, end="\n\n")

  if print_output:
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.savefig(os.path.join(constants.RESULT_PATH, 'auc.png'), facecolor='white', bbox_inches='tight')
    plt.show()

  if return_output:
    return sensitivity, specificity, f1_score, auc
