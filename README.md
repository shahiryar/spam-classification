# Spam Classifier using Transfer Learning with BERT

This repository contains a Python notebook, 'Spam Classification.ipynb', which implements a spam classifier using transfer learning with the BERT (Bidirectional Encoder Representations from Transformers) model. The dataset used for training and evaluation is provided in the 'spam_dataset.csv' file.

## Overview

The goal of this project is to build a spam classifier that can distinguish between spam and non-spam (ham) text messages. The model utilizes transfer learning, leveraging a pre-trained BERT model with the Hugging Face Transformers library.

## Requirements

- Python 3.x
- PyTorch
- Hugging Face Transformers
- pandas
- scikit-learn

You can install the required dependencies using the following command:

```bash
pip install torch transformers pandas scikit-learn
```

## Dataset

The dataset, 'spam_dataset.csv', contains labeled text messages, where the 'label' column indicates whether a message is spam (1) or not spam (0). The dataset is split into training and testing sets for model training and evaluation.

## Model Architecture

The spam classifier model is implemented using PyTorch and Hugging Face Transformers. The BERT model is frozen, and two fully connected layers are added on top of it. The final layer is a softmax layer with a single output, providing the probability of a message being spam.

The notebook covers the following key steps:
1. Loading and preprocessing the dataset
2. Loading the pre-trained BERT model from Hugging Face Transformers
3. Defining the model architecture with frozen BERT layers and additional fully connected layers
4. Training the model on the dataset
5. Evaluating the model's performance on the test set

# Model Evaluation
              precision    recall  f1-score   support

           0       0.98      0.89      0.93       483
           1       0.55      0.88      0.67        75

    accuracy                           0.89       558
   macro avg       0.76      0.88      0.80       558
weighted avg       0.92      0.89      0.90       558

## Confusion Matrix
[[428  55]
 [  9  66]]
 
## Usage
Open the notebook in Colab; make sure to use the GPU in the runtime settings

Feel free to customize the notebook for your specific use case, such as experimenting with hyperparameters or using different pre-trained BERT models.
