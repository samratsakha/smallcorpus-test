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
model = pickle.load(open('decision_1_model.pkl','rb'))



# generate url for mobiles
def getUrl(argument_1):
    switcher = {

        '4': "https://gadgets.ndtv.com/apple-iphone-4-765/user-reviews",
        '4S': "https://gadgets.ndtv.com/apple-iphone-4s-768/user-reviews",
        '5': "https://gadgets.ndtv.com/apple-iphone-5-771/user-reviews",
        '5S': "https://gadgets.ndtv.com/apple-iphone-5s-1028/user-reviews",
        '5C': "https://gadgets.ndtv.com/apple-iphone-5c-1027/user-reviews",

        'SE': "https://gadgets.ndtv.com/apple-iphone-se-3393/user-reviews",
        '6': "https://gadgets.ndtv.com/apple-iphone-6-1973/user-reviews",
        '6S': "https://gadgets.ndtv.com/apple-iphone-6s-2952/user-reviews",
        '6PLUS': "https://gadgets.ndtv.com/apple-iphone-6-plus-1974/user-reviews",
        '6SPLUS': "https://gadgets.ndtv.com/apple-iphone-6s-plus-2955/user-reviews",

        '7': "https://gadgets.ndtv.com/apple-iphone-7-3766/user-reviews",
        '7PLUS': "https://gadgets.ndtv.com/apple-iphone-7-plus-3767/user-reviews",
        '8': "https://gadgets.ndtv.com/apple-iphone-8-4260/user-reviews",
        '8PLUS': "https://gadgets.ndtv.com/apple-iphone-8-plus-4259/user-reviews",
        'X': "https://gadgets.ndtv.com/apple-iphone-x-4258/user-reviews",

        'XS': "https://gadgets.ndtv.com/apple-iphone-xs-5645/user-reviews",
        'XSMAX': "https://gadgets.ndtv.com/apple-iphone-xs-max-5646/user-reviews",
        'XR': "https://gadgets.ndtv.com/apple-iphone-xr-5647/user-reviews",
        '11': "https://gadgets.ndtv.com/apple-iphone-11-price-in-india-91110/user-reviews",
        '11PRO': "https://gadgets.ndtv.com/apple-iphone-11-pro-price-in-india-91112/user-reviews",

        '11PROMAX': "https://gadgets.ndtv.com/apple-iphone-11-pro-max-price-in-india-91111/user-reviews",
        '12': "https://gadgets.ndtv.com/iphone-12-price-in-india-97670/user-reviews",
        '12MINI': "https://gadgets.ndtv.com/iphone-12-mini-price-in-india-97685/user-reviews",
        '12PRO': "https://gadgets.ndtv.com/iphone-12-pro-price-in-india-97687/user-reviews",
        '12PROMAX': "https://gadgets.ndtv.com/iphone-12-pro-max-price-in-india-97686/user-reviews",
        'SE2020': "https://gadgets.ndtv.com/apple-iphone-se-2020-price-in-india-91195/user-reviews"

    }
    return switcher.get(argument_1)


# soup web scrapper
def scrap_reviews(mob_name):
    url = getUrl(mob_name)
    r = requests.get(url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    review_html = soup.find_all('div',class_="_cmttxt _wwrap")
    return review_html




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



# mobile id 
def getmobid(passed_argument):
    switcher_2 = {
    
        "4":1,   
        "4S":2,
        "5":3,
        "5S":4,
        "5C":5,

        "SE":6,
        "6":7,
        "6S":8,
        "6PLUS":9,
        "6SPLUS":10,

        "7":11,
        "7PLUS":12,
        "8":13,
        "8PLUS":14,
        "X":15,

        "XS":16,
        "XSMAX":17,
        "XR":18,
        "11":19,
        "11PRO":20,

        "11PROMAX":21,
        "12":22,
        "12MINI":23,
        "12PRO":24,
        "12PROMAX":25,
        "SE2020":14

    } 
    return switcher_2.get(passed_argument)


# coverts Y/N to 1 and 0
def string_to_binary(pass_argument):
    if(pass_argument=="YES"):
        pass_argument=1
    else:
        pass_argument=0
    return pass_argument




# function for predicting prices of iphone
def price_predicter(mob_model,vart,pd,sd,hd,bt,kt):
    mob_id = getmobid(mob_model)
    pd = string_to_binary(pd)
    sd = string_to_binary(sd)
    hd = string_to_binary(hd)
    kt = string_to_binary(kt)

    prediction_price = model.predict([[ mob_id , vart , pd , sd , hd , bt , kt ]])

    return prediction_price




# flask templates
@app.route('/',methods=['GET'])
def Home():
    return render_template('home.html')


# home to buy 
@app.route("/buy", methods=['POST'])
def gotobuy():
    return render_template('index.html')



# evaluating sentiments
@app.route("/review", methods=['POST'])
def review_this():
    pos,neg,neut=0,0,0
    rt_1,rt_2,rt_3,rt_4,rt_5=0,0,0,0,0
    sentiment_review,sentiment_rate=0,0
    count=0 

    review_text=""

    if request.method == 'POST':

        pos,neg,neut=0,0,0
        sentiment_review,sentiment_rate=0,0
        count=0 

        model_mob = request.form['model_mob']
        variant = request.form['variant']
        model_mob = model_mob.replace(" ","")

        predicted_price = price_predicter(model_mob,variant,"NO","NO","NO",85,"YES")
        predicted_price = round(predicted_price[0],0)
        predicted_price = predicted_price.astype(int)

        review_html = scrap_reviews(model_mob)

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

        
        
        return render_template('review.html',classification_text="Total Review {}".format(count) , 
        prediction_text="Positive Review is {}".format(pos), 
        prediction_text_1="Neutral Review is {}".format(neut), 
        prediction_text_2="Negative Review is {}".format(neg),
        classification_text_1="5 Star Rating {}".format(rt_5),
        classification_text_2="4 Star Rating {}".format(rt_4),
        classification_text_3="3 Star Rating {}".format(rt_3),
        classification_text_4="2 Star Rating {}".format(rt_2),
        classification_text_5="1 Star Rating {}".format(rt_1),
        prediction_price_mob="Price of mobile is {}".format(predicted_price))
        
    else:

        return render_template('review.html')



if __name__=="__main__":
    app.run(debug=True)
