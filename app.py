# import libraries
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


# load the models
cv_2 = pickle.load(open("cv_2_mobile.pkl","rb"))
cv_3 = pickle.load(open("cv_2_mobile_rating.pkl","rb"))
loaded_model = pickle.load(open('model_mobile.pkl',"rb"))
loaded_model_2 = pickle.load(open('model_mobile_rating.pkl',"rb"))


# soup web scrapper
url = "https://gadgets.ndtv.com/apple-iphone-11-price-in-india-91110/user-reviews"
r = requests.get(url)
htmlContent = r.content
soup = BeautifulSoup(htmlContent, 'html.parser')
review_html = soup.find_all('div',class_="_cmttxt _wwrap")

# flask templates
@app.route('/',methods=['GET'])
def Home():
    return render_template('home.html')


# sentiment analysis function 
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


# decision made on sentiment text
def decision_maker(final_review , rate_review):
    rate_it = np.round( rate_review, decimals=2)

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

    return output,final_rate




@app.route("/buy", methods=['POST'])
def gotobuy():
    return render_template('index.html')


# evaluating sentiments


@app.route("/classify", methods=['POST'])
def classify():
    pos,neg,neut=0,0,0
    rt_1,rt_2,rt_3,rt_4,rt_5=0,0,0,0,0
    sentiment_review,sentiment_rate=0,0
    count=0 

    review_text=""

    if request.method == 'POST':

        pos,neg,neut=0,0,0
        sentiment_review,sentiment_rate=0,0
        count=0 

        review_text=request.form['enter_review']

        for i in review_html:
           text = i.get_text()
           count+=1
           sentiment_review,sentiment_rate=new_review(str(text))
           get_output,get_rate=decision_maker(sentiment_review,sentiment_rate)

           if get_output=="Positive":
               pos+=1
           elif get_output=="Neutral":
               neut+=1
           elif get_output=="Negative":
               neg+=1
               
           if get_rate==1:
                rt_1+=1
           elif get_rate==2:
                rt_2+=1
           elif get_rate==3:
                rt_3+=1
           elif get_rate==4:
                rt_4+=1
           elif get_rate==5:
                rt_5+=1

        
        
        return render_template('index.html',classification_text="Total Review {}".format(count) , 
        prediction_text="Positive Review is {}".format(pos), 
        prediction_text_1="Neutral Review is {}".format(neut), 
        prediction_text_2="Negative Review is {}".format(neg),
        classification_text_1="5 Star Rating {}".format(rt_5),
        classification_text_2="4 Star Rating {}".format(rt_4),
        classification_text_3="3 Star Rating {}".format(rt_3),
        classification_text_4="2 Star Rating {}".format(rt_2),
        classification_text_5="1 Star Rating {}".format(rt_1))
        
    else:

        return render_template('index.html')



if __name__=="__main__":
    app.run(debug=True)
