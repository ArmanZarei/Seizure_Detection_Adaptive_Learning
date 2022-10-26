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

- Offline and Online train
    - Offline train
        - Before conducted on the patient
    - Loss and accuracy plot
    - Evaluation
        - Confusion Matrix
        - Accuracy
        - Sensitivity
        - Specificity
        - F1-Score
        - AUC
        - ROC
    - Adaptive learning 
    - Accuracy and sensitivity over time plot

    Example:
    ![](images/offline_and_online_train.png)
