# Seizure Detection using Adaptive Learning

Tutorial: [Link](tutorial.ipynb)

## Modules

- Dataset + Feature Extraction
    - Download a specific patient files
    - Extract windows and labels of patient (from files)
    - Sample from windows (fro trainset)
        - Retain non-seizure windows adjacent to seizure windows
    - Feature extraction 
        - Spectral power and line length
    - EEGDataset
        - Pytorch dataset wrapper
    
    Example:
    ![](images/dataset_usage.png)

