# Intrusion Detection System for IoT and IIoT Networks

This project focuses on developing intrusion detection systems (IDS) for Internet of Things (IoT) and Industrial Internet of Things (IIoT) networks using machine learning and deep learning techniques. It includes the implementation and evaluation of IDS models using two datasets: CoAP-DDoS and Edge-IIoT.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [CoAP-DoS Dataset](#coap-dos-dataset)
- [Edge-IIoT Dataset](#edge-iiot-dataset)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In recent years, the proliferation of IoT and IIoT devices has led to an increased risk of cyberattacks targeting these networks. Intrusion detection systems play a crucial role in identifying and mitigating such attacks. This project aims to develop effective IDS models tailored for IoT and IIoT environments.

## Datasets

The project utilizes two datasets for training and evaluating the IDS models:

- **<a href="https://www.kaggle.com/datasets/jaredalanmathews/coapddos">CoAP-DoS Dataset</a>**: A dataset containing network traffic data related to CoAP-based DDoS attacks. <a href="https://ieeexplore.ieee.org/document/9845285">Original Publication</a>.
- **<a href="https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot">Edge-IIoT Dataset**: A dataset comprising network traffic data from Edge-IIoT environments, including various types of attacks and normal traffic. <a href="https://ieee-dataport.org/documents/edge-iiotset-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications">Original Publicaiton</a>.

## CoAP-DoS Dataset

### Description

The CoAP-DoS dataset consists of network traffic data captured during CoAP-based DDoS attacks. It includes features such as packet headers, payload information, and timestamps.

### Preprocessing

Preprocessing steps applied to the CoAP-DoS dataset include median filtering, standard deviation-based filtering, and normalization. These steps help in cleaning the data and preparing it for model training.

### Model Architecture

The IDS model architecture for the CoAP-DoS dataset consists of convolutional and recurrent neural network layers. These layers are designed to extract relevant features from the input data and make predictions based on them.

### Training and Evaluation

The model is trained using the training data from the CoAP-DoS dataset and evaluated using the test data. Training involves optimizing the model's parameters using the Adam optimizer and minimizing the sparse categorical crossentropy loss. The model's performance is evaluated based on accuracy metrics.

## Edge-IIoT Dataset

### Description

The Edge-IIoT dataset comprises network traffic data collected from Edge-IIoT environments, including various types of attacks and normal traffic patterns. It contains features related to network protocols, communication patterns, and attack types.

### Preprocessing

Preprocessing of the Edge-IIoT dataset involves encoding categorical features, scaling numerical features, and reshaping the data for model compatibility. These preprocessing steps ensure that the data is in a suitable format for training the IDS model.

### Model Architecture

The IDS model architecture for the Edge-IIoT dataset includes convolutional, pooling, and recurrent layers followed by dense layers for classification. This architecture is designed to capture temporal and spatial dependencies in the input data and make accurate predictions.

### Training and Evaluation

The model is trained using the preprocessed training data from the Edge-IIoT dataset and evaluated using the test data. Training involves optimizing the model's parameters using the Adam optimizer and minimizing the categorical crossentropy loss. Model performance is assessed using accuracy metrics and confusion matrices.

## Usage

To use the project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies mentioned in the `requirements.txt` file.
3. Run the provided Jupyter notebooks or Python scripts to train and evaluate the IDS models.
4. Experiment with different hyperparameters and architectures to improve model performance.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

© 2024 ALI BAYANI
