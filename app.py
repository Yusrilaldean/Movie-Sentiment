from flask import Flask, render_template, url_for, jsonify, send_file, request
app = Flask(__name__)

import io
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def readData():
    return pd.read_excel('dataset.xlsx')

def tokenizeData(dataFrame):
    tweets = dataFrame['Tweets'].values
    valueOfTweets = dataFrame['Value']
    token_test = Tokenizer()
    token_test.fit_on_texts(tweets)
    seq_data_tweet_test = Tokenizer.texts_to_sequences(token_test, tweets)
    enc_data_tweet_test = Tokenizer.sequences_to_matrix(token_test, seq_data_tweet_test, mode="tfidf")
    x_test = enc_data_tweet_test
    y_test = valueOfTweets
    return x_test, y_test

def prediction(x, y):
    clf = pickle.load(open('model.sav', 'rb'))
    clf.fit(x, y)
    pred = clf.predict(x)
    classReport = classification_report(y, pred, output_dict=True)
    return classReport['accuracy']

@app.route('/')
@app.route('/index')
def index():
    testDataX, testDataY = tokenizeData(readData())
    predictionResult = prediction(testDataX, testDataY)

    return render_template('index.html', predictionResultData=predictionResult)

if __name__ == '__main__':
    app.run(debug=True)
