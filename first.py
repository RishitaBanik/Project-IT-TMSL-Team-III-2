import numpy as np
import flask
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import joblib
import pickle
import os
import sklearn
import urllib
import newspaper
from newspaper import fulltext
from newspaper import Article
from bs4 import BeautifulSoup
import requests
import nltk
from time import sleep
from newspaper.article import ArticleException, ArticleDownloadState

app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__,template_folder='templates')

with open('model.pickle', 'rb') as handle:
   model = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    url=request.get_data(as_text=True)[5:]
    url=urllib.parse.unquote(url)
    #url='https://www.electronics-tutorials.ws/accircuits/ac-waveform.html'
    article=Article(url)
    article.download()
    article.parse()
    nltk.download('punkt')
    article.nlp()
    news=article.summary
    pred=model.predict([news])
    return render_template('index.html', prediction_text='THE NEWS IS "{}"'.format(pred[0]))

if __name__ == "__main__":
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)
