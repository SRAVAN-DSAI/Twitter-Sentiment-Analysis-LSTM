# Twitter-Sentiment-Analysis-LSTM

This repository contains code for building, training, and deploying a sentiment analysis model using Long Short-Term Memory (LSTM) networks. The model 

predicts sentiment based on tweets, classifying them into five categories: 

**0 (negative)**,
**1 (neutral)**,
**2 (positive)**,
**3 (very positive)**, or 
**4 (very negative)**.

## Requirements

To run the code, ensure you have the following dependencies installed:
- Python 
- TensorFlow 
- Numpy
- Pandas
- Scikit-learn
- NLTK
- Streamlit
- Pickle

You can install the necessary libraries by running:
```
pip install tensorflow numpy pandas scikit-learn nltk streamlit pickle-mixin
```
Files

training.1600000.processed.noemoticon.csv: The dataset containing tweets and their corresponding sentiment labels.

sentimental_model.h5: The trained LSTM model (saved after training).

tokenizer.pickle: The saved tokenizer used to preprocess text data before model inference.

How It Works

1. Data Loading & Preprocessing
   
The dataset is loaded and preprocessed by:

Converting the tweet text to lowercase.

Removing non-alphabetic characters.

Tokenizing the text into individual words.

Removing stop words (commonly used words like "the", "and", etc.).

Lemmatizing and stemming the words to ensure uniformity.

2. Tokenization & Padding
   
The text data is tokenized using Keras's Tokenizer class and then padded to ensure that all sequences are of the same length. This step converts text 

into numerical sequences that the model can understand.

3. Model Architecture
   
A Sequential model with LSTM layers is built:

An Embedding Layer converts input sequences into word vectors.

Two LSTM Layers capture sequential patterns in the text.

Dense Layers and a Dropout Layer are added for classification and to prevent overfitting.

4. Model Training
   
The model is trained using the training data and validated with a validation split to track performance. The trained model is then saved as a .h5 file.

5. Saving the Model & Tokenizer
   
The model and tokenizer are saved after training. This allows the model to be reused later for inference without retraining.

6. Streamlit Deployment

The trained model and tokenizer are loaded into a Streamlit app, which provides an interactive interface for users to enter tweet text and get real-

time sentiment predictions.

How to Use

Training the Model:

Use the provided dataset and code to train the LSTM model. The code will process the data, train the model, and save it as sentimental_model.h5 along 

with the tokenizer (tokenizer.pickle).

Deploying with Streamlit:

Once the model is trained, you can deploy it for real-time predictions using Streamlit. The web interface allows users to input tweet text and predict 

its sentiment.

To run the Streamlit app, execute the following command in your terminal:
```
streamlit run app.py
```
This will open a web page where you can enter tweet text and see the predicted sentiment
