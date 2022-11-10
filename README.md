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

## Results and comparison with SOUL paper

### Setup
- Dataset: CHB-MIT Scalp EEG Database
- Patient number: `8`
- Dataset Partitioning
    - Train: `26%`
    - Validation: `5%`
    - Test: `69%`


### Linear Regression

- Offline train 
    ![](images/comparison/train_log.png)

    ![](images/comparison/train_loss_acc_plot.png)

- Parameter tunning
    ![](images/comparison/parameter_tuning.png)

- Evaluation
    ![](images/comparison/evaluation.png)

- Accuracy and Sensitivity over time (Adaptive Learning Phase)
    ![](images/comparison/acc_sen_over_time.PNG)

### MLP 

![](images/comparison/mlp_model.png)

**Note**: Only the last layer will be re-trained during adaptive learning phase

- Offline train 
    ![](images/comparison/mlp_train_log.png)

    ![](images/comparison/mlp_train_loss_acc_plot.png)

- Parameter tunning
    ![](images/comparison/mlp_parameter_tuning.png)

- Evaluation
    ![](images/comparison/mlp_evaluation.png)

- Accuracy and Sensitivity over time (Adaptive Learning Phase)
    ![](images/comparison/mlp_acc_sen_over_time.PNG)

### CNN

![](images/comparison/cnn_model.png)

**Note**: Only the last (linear) layer will be re-trained during adaptive learning phase

- Offline train 
    ![](images/comparison/cnn_train_log.png)

    ![](images/comparison/cnn_train_loss_acc_plot.png)

- Parameter tunning
    ![](images/comparison/cnn_parameter_tuning.png)

- Evaluation
    ![](images/comparison/cnn_evaluation.png)

- Accuracy and Sensitivity over time (Adaptive Learning Phase)
    ![](images/comparison/cnn_acc_sen_over_time.PNG)
