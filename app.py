# import libraries
import flask
from flask import Flask, render_template, request ,url_for
import gspread
from oauth2client.service_account import ServiceAccountCredentials
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
# scrap user reviews
def scrap_reviews(mob_name):
    url = getUrl(mob_name)
    r = requests.get(url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    review_html = soup.find_all('div',class_="_cmttxt _wwrap")
    return review_html

# scrap tech reviews
def scrap_tech_reviews(mob_name):
    url = getUrl(mob_name)
    new_url = url.replace("/user-reviews" , "")
    r = requests.get(new_url)
    htmlContent = r.content
    soup = BeautifulSoup(htmlContent, 'html.parser')
    tech_html = soup.find_all('i')

    list_class = []

    class_list = set()
    for i in tech_html:
        if i.has_attr("class"):
            x = str(i)
            list_class.append(x)

    string =""
    count=0
    print(new_url)
    for i in list_class:
        if (i.find('_sp r6')!=-1):
            string+='6 ' 
            count+=1
        elif (i.find('_sp r7')!=-1):
            string+='7 '
            count+=1
        elif (i.find('_sp r8')!=-1):
            string+='8 ' 
            count+=1
        elif (i.find('_sp r9')!=-1):
            string+='9 '
            count+=1
        elif (i.find('_sp r10')!=-1):
            string+='10 '
            count+=1
        elif(count==8):
            break
    
    return string 


# calculate resale value
def calculate_price(text):
    price = 0
    if (text<=10000):
        price = (text*70)//100
    elif (text>10000 and text<=20000):
        price = (text*80)//100
    elif (text>20000 and text<=35000):
        price = (text*85)//100
    elif (text>35000 and text<=50000):
        price = (text*88)//100
    elif (text>50000 and text<=80000):
        price = (text*90)//100
    elif (text>80000):
        price = (text*93)//100

    return price
  


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
    output=""
    final_rate = np.round( rate_review , decimals=2 )

    if rate_it>4.3:
        final_rate = 5
    elif rate_it==3.72:
        final_rate = 3
    elif rate_it>3.5 and rate_it<=4.3:
        final_rate = 4
    elif rate_it>2.5 and rate_it<=3.5:
        final_rate = 3
    elif rate_it>1.5 and rate_it<=2.5:
        final_rate = 2
    elif rate_it<=1.5:
        final_rate = 1

    if final_review==0 or final_rate<3:
        output="Negative"
        if final_rate>=3:
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



#############################################     Buying Section     ###############################################

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

        
        
        return render_template('review.html',classification_text=count , 
        prediction_text=pos, 
        prediction_text_1=neut, 
        prediction_text_2=neg,
        classification_text_1=rt_5,
        classification_text_2=rt_4,
        classification_text_3=rt_3,
        classification_text_4=rt_2,
        classification_text_5=rt_1,
        prediction_price_mob=predicted_price,
        mobile_model_name=model_mob,
        mob_variant=variant)
        
    else:

        return render_template('review.html')


# scrap the tech ratings
@app.route("/tech_review", methods=['POST'])
def tech_review():
    if request.method == 'POST':
        model_of_mob = request.form['model_mob_name']
        variant_of_mob = request.form['model_mob_variant']
        model_of_mob = model_of_mob.replace(" ","")
        variant_of_mob = variant_of_mob.replace(" ","")

        get_tech = scrap_tech_reviews(model_of_mob)

        return render_template('tech_review.html',tech_reviews=get_tech,
        model_of_the_mob=model_of_mob,variant_of_the_mob=variant_of_mob)

    else:

        return render_template('tech_review.html')


# check availability of iphone from database
@app.route("/buy_iphone", methods=['POST'])
def buy_iphone():
    if request.method == 'POST':
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials_iphone_available_list.json',scope)

        client = gspread.authorize(credentials)
        sheet = client.open("available_iphones_list").sheet1

        models = [item for item in sheet.col_values(1) if item]
        variants = [item for item in sheet.col_values(2) if item]
        colors = [item for item in sheet.col_values(3) if item]
        condition = [item for item in sheet.col_values(4) if item]
        idnum = [item for item in sheet.col_values(5) if item]

        model_of_mob = request.form['model_mob_name']
        variant_of_mob = request.form['model_mob_variant']
        model_of_mob = model_of_mob.replace(" ","")
        variant_of_mob = variant_of_mob.replace(" ","")

        avail = 0
        string = ""
        pass_id = ""
        for i in range(len(models)):
            if(models[i]==model_of_mob and variants[i]==variant_of_mob):
                string += (models[i]+" "+variants[i]+"GB "+colors[i]+" "+condition[i]+"|")
                pass_id += idnum[i]+"|"
                avail += 1

        return render_template('availability.html',availability=avail,get_string=string,
        get_id=pass_id,get_model=model_of_mob,variant_of_the_mob=variant_of_mob)

    else:

        return render_template('availability.html')



#store the choosed iphone to buy to database 
@app.route("/thanks", methods=['POST'])
def store_buyer_iphone():
    if request.method == 'POST':
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials_iphone_available_list.json',scope)

        client = gspread.authorize(credentials)
        sheet = client.open("available_iphones_list").get_worksheet(1)
        rows = [item for item in sheet.col_values(1) if item]

        model_selected = request.form['selected_iphone']
        mob_num = request.form['mob_num']
        feedback = request.form['feedback']

        sentiment_review,sentiment_rate=new_review(str(feedback))
        get_output,get_rate=decision_maker(sentiment_review,sentiment_rate)
        
        len_row = len(rows)+1

        sheet.update_cell(len_row,1,model_selected)
        sheet.update_cell(len_row,2,mob_num)
        sheet.update_cell(len_row,3,get_rate)
        sheet.update_cell(len_row,4,get_output)
        sheet.update_cell(len_row,5,feedback)

        return render_template("thanks.html",from_section="BUY")

    else:

        return render_template("thanks.html",from_section="BUY")


#redirect to home page after thanks page
@app.route("/home", methods=['POST'])
def go_to_home():
    if request.method == 'POST':

        return render_template("home.html")

    else:

        return render_template("home.html")







#############################################     Selling Section     ###############################################



# home to sell

@app.route("/sell", methods=['POST'])
def gotosell():
    return render_template('index2.html') 



# calculate resale value 
@app.route("/resale_value", methods=['POST'])
def get_resale_value():
    if request.method == 'POST':
        model_mob_sell=request.form['model_mob_sell']
        variant_sell=int(request.form['variant_sell'])
        pd=request.form['physical_damage']
        sd=request.form['software_issues']
        hd=request.form['hardware_issues']
        battery=int(request.form['battery'])
        kit=request.form['kit']

        resales_value = price_predicter(model_mob_sell,variant_sell,pd,sd,hd,battery,kit)

        resales_value = round(resales_value[0],0)
        resales_value = resales_value.astype(int)

        resales_value = calculate_price(resales_value)

        return render_template('sell.html',prediction_text=resales_value,
        physical_damage = pd,
        software_damage = sd,
        hardware_damage = hd,
        battery_percent = battery,
        kit_availability = kit,
        mobile_model=model_mob_sell,
        variant_=variant_sell)

    else:

        return render_template('sell.html')



# resale_value to sell_iphone
@app.route("/sell_iphone", methods=['POST'])
def sell_iphone():
    if request.method == 'POST':
        mob_details = request.form['pass_model_details']

        return render_template('sell_iphone.html',mobile_details=mob_details)

    else:

        return render_template('sell.html')


# store all selling details to databse
@app.route("/sell_this_iphone", methods=['POST'])
def sell_this_iphone():
    if request.method == 'POST':
        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials_iphone_available_list.json',scope)

        client = gspread.authorize(credentials)
        sheet = client.open("available_iphones_list").get_worksheet(2)
        rows = [item for item in sheet.col_values(1) if item]
        len_row = len(rows)+1

        pd_details = request.form['pd_details']
        sd_details = request.form['sd_details']
        hd_details = request.form['hd_details']
        kit_details = request.form['kit_details']
        age_details = request.form['age_details']
        mob_details = request.form['mob_details']
        descp_details = request.form['descp_details']
        mob_num = request.form['mob_num']
        feedback = request.form['feedback']

        mob_details = mob_details.split("|")
        model_variant = mob_details[1]+" "+mob_details[2]+"GB"
        if(kit_details=="NO"):
            kit_details="Available"

        
        sentiment_review_2,sentiment_rate_2=new_review(str(descp_details))
        get_output_2,get_rate_2=decision_maker(sentiment_review_2,sentiment_rate_2)
        descp_rate = str(get_output_2)+" "+str(get_rate_2)

        sentiment_review,sentiment_rate=new_review(str(feedback))
        get_output,get_rate=decision_maker(sentiment_review,sentiment_rate)
        feed_rate = str(get_output)+" "+str(get_rate)

        sheet.update_cell(len_row,1,model_variant)
        sheet.update_cell(len_row,2,mob_details[8])
        sheet.update_cell(len_row,3,mob_details[0])
        sheet.update_cell(len_row,4,pd_details)
        sheet.update_cell(len_row,5,sd_details)
        sheet.update_cell(len_row,6,hd_details)
        sheet.update_cell(len_row,7,kit_details)
        sheet.update_cell(len_row,8,mob_details[6])
        sheet.update_cell(len_row,9,age_details)
        sheet.update_cell(len_row,10,descp_details)
        sheet.update_cell(len_row,11,descp_rate)
        sheet.update_cell(len_row,12,feedback)
        sheet.update_cell(len_row,13,feed_rate)
        sheet.update_cell(len_row,14,mob_num)



        return render_template('thanks.html',from_section="SELL")
    
    else:

        return render_template('thanks.html',from_section="SELL")




#############################################     Reviewing Section     ###############################################


# home to review iphone 
@app.route("/review_iphone", methods=['POST'])
def go_to_review_our_iphone():
    if request.method == 'POST':

        return render_template('iphone_review.html')

    else:

        return render_template('home.html')


# home to complaint iphone 
@app.route("/complaint_iphone", methods=['POST'])
def go_to_complaint_our_iphone():
    if request.method == 'POST':

        return render_template('complaint.html')

    else:

        return render_template('home.html')



# review our iphone to database 
@app.route("/thank_you", methods=['POST'])
def review_our_iphone():
    if request.method == 'POST':

        mob_model = request.form['model_mob']
        mob_variant = int(request.form['variant'])
        mob_variant = str(mob_variant) + "GB"
        imei = request.form['imei']
        imei = imei.replace(" ","")
        review_texts = request.form['review_texts']
        mobile_number = int(request.form['mobile_number'])

        sentiment_review,sentiment_rate=new_review(str(review_texts))
        get_output,get_rate=decision_maker(sentiment_review,sentiment_rate)


        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials_iphone_available_list.json',scope)

        client = gspread.authorize(credentials)
        sheet = client.open("available_iphones_list").get_worksheet(3)
        row_imei = [item for item in sheet.col_values(1) if item]
        row_info = [item for item in sheet.col_values(2) if item]
        len_row = len(row_info)+1

        models = []
        variants = []
        flag = 0


        for i in row_info:
            splitted = i.split()
            models.append(splitted[0])
            variants.append(splitted[1])


        for i in range(len_row-1):
            if(row_imei[i].replace(" ","")==imei):
                if(mob_model==models[i] and mob_variant==variants[i]):
                    sheet.update_cell(i+1,3,get_output)
                    sheet.update_cell(i+1,4,get_rate)
                    sheet.update_cell(i+1,5,review_texts)
                    sheet.update_cell(i+1,6,mobile_number)
                    flag = 1
                
                
        if flag==1:
            return render_template('thanks.html',from_section="review_1")
        else:
            return render_template('thanks.html',from_section="review_0")


    else:

        return render_template('home.html')




# iphone complaints 
@app.route("/thankyou", methods=['POST'])
def complaint_iphone():
    if request.method == 'POST':

        mob_model = request.form['model_mob']
        mob_variant = int(request.form['variant'])
        mob_variant = str(mob_variant) + "GB"
        imei = request.form['imei']
        imei = imei.replace(" ","")
        complaint = request.form['review_texts']
        mobile_number = int(request.form['mobile_number'])

        scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
        credentials = ServiceAccountCredentials.from_json_keyfile_name('credentials_iphone_available_list.json',scope)

        client = gspread.authorize(credentials)
        sheet = client.open("available_iphones_list").get_worksheet(3)
        row_imei = [item for item in sheet.col_values(1) if item]
        row_info = [item for item in sheet.col_values(2) if item]
        len_row = len(row_info)+1

        models = []
        variants = []
        flag = 0

        for i in row_info:
            splitted = i.split()
            models.append(splitted[0])
            variants.append(splitted[1])


        for i in range(len_row-1):
            if(row_imei[i].replace(" ","")==imei):
                if(mob_model==models[i] and mob_variant==variants[i]):
                    sheet.update_cell(i+1,10,complaint)
                    sheet.update_cell(i+1,11,mobile_number)
                    flag = 1

        if flag==1:
            return render_template('thanks.html',from_section="complaint_1")
        else:
            return render_template('thanks.html',from_section="complaint_0")


    else:

        return render_template('home.html')






if __name__=="__main__":
    app.run(debug=True)
