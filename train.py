import numpy as np
from termcolor import colored
from datetime import datetime
import torch


def train(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs):
  """
    Trains (offline) a given model for a specific training dataset

    Parameters:
        model (torch.nn.Module): The model to be trained
        criterion (torch.nn.CrossEntropyLoss | ...): Loss function used for training
        optimizer (torch.optim.optimizer.SGD | ...): Optimizer used for training
        scheduler (torch.optim.lr_scheduler.StepLR | ...): Scheduler used for training
        train_loader (torch.utils.data.DataLoader): Batches of training data 
        val_loader (torch.utils.data.DataLoader | None): Batches of validation data (if given). 
        num_epochs (): Number of training epochs 

    Returns:
        train_loss_arr (list): List of training dataset loss at each epoch
        val_loss_arr (list): List of validation dataset loss at each epoch
        [if val_loader != None] train_acc_arr (list): List of training dataset accuracy at each epoch
        [if val_loader != None] val_acc_arr (list): List of validation dataset accuracy at each epoch
    """

  train_loss_arr, val_loss_arr = [], []
  train_acc_arr, val_acc_arr = [], []
  for epoch in range(num_epochs):
    train_loss, val_loss = .0, .0
    train_acc, val_acc = .0, .0
    train_confusion_matrix, val_confusion_matrix = np.zeros((2, 2)), np.zeros((2, 2)) # 0 -> label , 1 -> pred    =====>    0,0 -> TN  -  0,1 -> FP  -  1,0 -> FN  -  1,1 -> TP

    model.train()
    for X, Y in train_loader:
      optimizer.zero_grad()
      outputs = model(X.type(torch.FloatTensor)) 
      loss = criterion(outputs, Y)
      train_loss += loss.detach() * X.size(0)
      output_labels = torch.max(outputs, axis=1)[1]
      train_acc += torch.sum(output_labels == Y).item()

      for l_Y in range(2):
        for l_output in range(2):
          train_confusion_matrix[l_Y][l_output] += torch.sum((Y == l_Y) & (output_labels == l_output)).item()
          
      loss.backward()
      optimizer.step()
    scheduler.step()

    if val_loader is not None:
      model.eval()
      for X, Y in val_loader:
        outputs = model(X.type(torch.FloatTensor)) 
        loss = criterion(outputs, Y)
        val_loss += loss.item() * X.size(0)
        output_labels = torch.max(outputs, axis=1)[1]
        val_acc += torch.sum(output_labels == Y).item()
        for l_Y in range(2):
          for l_output in range(2):
            val_confusion_matrix[l_Y][l_output] += torch.sum((Y == l_Y) & (output_labels == l_output)).item()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)
    train_loss_arr.append(train_loss)
    train_acc_arr.append(train_acc)
    train_sensitivity = train_confusion_matrix[1, 1]/np.sum(train_confusion_matrix, axis=1)[1]
    train_specificity = train_confusion_matrix[0, 0]/np.sum(train_confusion_matrix, axis=1)[0]

    if val_loader is not None:
      val_loss /= len(val_loader.dataset)
      val_acc /= len(val_loader.dataset)
      val_acc_arr.append(val_acc)
      val_loss_arr.append(val_loss)
      val_sensitivity = val_confusion_matrix[1, 1]/np.sum(val_confusion_matrix, axis=1)[1]
      val_specificity = val_confusion_matrix[0, 0]/np.sum(val_confusion_matrix, axis=1)[0]

    if val_loader is not None:
      print(f"[Epoch {epoch:3d}]",
            f"[{datetime.now().strftime('%H:%M:%S')}]\t",
            colored(f"Train Loss: {train_loss:.4f}", 'red'),
            colored(f"Train Acc: {train_acc:.2f}", 'blue'),
            colored(f"Train Sens: {train_sensitivity:.2f}", 'green'),
            colored(f"Train Spec: {train_specificity:.2f}\t\t", 'cyan'),
            colored(f"Val Loss: {val_loss:.4f}", 'red'),
            colored(f"Val Acc: {val_acc:.2f}", 'blue'),
            colored(f"Val Sens: {val_sensitivity:.2f}", 'green'),
            colored(f"Val Spec: {val_specificity:.2f}\t", 'cyan'),
          )
    else:
      print(f"[Epoch {epoch:3d}]",
            f"[{datetime.now().strftime('%H:%M:%S')}]\t",
            colored(f"Train Loss: {train_loss:.4f}", 'red'),
            colored(f"Train Acc: {train_acc:.2f}", 'blue'),
            colored(f"Train Sens: {train_sensitivity:.2f}", 'green'),
            colored(f"Train Spec: {train_specificity:.2f}", 'cyan'),
          )

  if val_loader is not None:
    return train_loss_arr, val_loss_arr, train_acc_arr, val_acc_arr
  else:
    return train_loss_arr, train_acc_arr
