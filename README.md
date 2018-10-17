# Time-Series-Anomaly-Detection

** WIP **

Testing encoder-decoder architecture to detect anomalies in time series unlabeled* data.

Using the following data sets: ECG record 100 from MIT-BIH arrhythmia database [https://www.physionet.org/physiobank/database/mitdb/] http://courses.csail.mit.edu/18.337/2017/projects/morales_manuel/datasets/ecg_data/MIT-BIH-arrhythmia-database/.

Method:
  - LSTM based Autoencoder
  
This project was used as part of the validation process for a model architecture used to detect anomalies in time series data.

```* The ECG data is labeled but the anomaly detection is based on the reconstruction error produced by the autoencoder, and the labels are only used as part of the validation and not directly used for training the model.```

### Model Architecture
![alt text](https://github.com/Mysjkin/Time-Series-Anomaly-Detection/blob/master/output/figures/lstm-model-1.png)

### Data Exploration
![alt text](https://github.com/Mysjkin/Time-Series-Anomaly-Detection/blob/master/output/figures/data_and_anomalies.png)

### Results
![alt text](https://github.com/Mysjkin/Time-Series-Anomaly-Detection/blob/master/output/figures/actual_predicted.png)
