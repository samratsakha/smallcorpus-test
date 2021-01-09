import flask
from flask import Flask, render_template, request ,url_for
import jsonify
import requests
import bs4
import textblob
from bs4 import BeautifulSoup
from textblob import TextBlob
import sklearn
import numpy as np
import re
import joblib
import pickle     

app = Flask(__name__)


cv_2 = pickle.load(open("cv_2_mobile.pkl","rb"))
cv_3 = pickle.load(open("cv_2_mobile_rating.pkl","rb"))
loaded_model = pickle.load(open('model_mobile.pkl',"rb"))
loaded_model_2 = pickle.load(open('model_mobile_rating.pkl',"rb"))


@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


from sklearn.feature_extraction.text import TfidfVectorizer
def new_review(new_review):
    new_review = new_review
    new_review = re.sub('[^a-zA-Z]', ' ', new_review)
    new_review = new_review.lower()
    new_review = new_review.split()
    all_stopwords = new_review
    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv_2.transform(new_corpus).toarray()
    new_X_test_2 = cv_3.transform(new_corpus).toarray()
    pred = loaded_model.predict(new_X_test)
    pred_2 = loaded_model_2.predict(new_X_test_2)
    return pred , pred_2





@app.route("/classify", methods=['POST'])
def classify():

    review_text=""

    if request.method == 'POST':

        review_text=request.form['enter_review']
        final_review , rate_review = new_review(review_text)

        rate_it = x = np.round( rate_review, decimals=2)


        if rate_it>4.3:
            final_rate = 5
        elif rate_it==3.72:
            final_rate = 3
        elif rate_it>3.5 and rate_it<4.3:
            final_rate = 4
        elif rate_it>2.5 and rate_it<3.5:
            final_rate = 3
        elif rate_it>1.5 and rate_it<2.5:
            final_rate = 2
        elif rate_it<1.5:
            final_rate = 1

        if final_review==0 or final_rate<3:
            output="Negative"
            if final_rate>3:
                final_rate=2
        elif final_review == 1 and final_rate>3:
            output="Positive"
        elif final_review==1 and final_rate==3:
            output="Neutral"
        elif final_review==0 and final_rate==3:
            output="Negative"
        elif final_review==0 and final_rate<3:
            output="Negative"
        
        

         

        
        return render_template('index.html',classification_text="Your Review is {}".format(output) , prediction_text="Your Rating is {}".format(final_rate))
        

    else:

        return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)
