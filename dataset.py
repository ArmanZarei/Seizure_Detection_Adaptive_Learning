from torch.utils.data import Dataset


class EEGDataset(Dataset):
  """
    Dataset containing windows and labels of EEG signals
  """
  
  def __init__(self, windows, labels, transform=None):
    self.windows = windows
    self.labels = labels
    self.transform = transform
  
  def __len__(self):
    return self.labels.shape[0]
  
  def __getitem__(self, idx):
    x = self.windows[idx]
    if self.transform:
      x = self.transform(self.windows[idx])

    return x, self.labels[idx]
