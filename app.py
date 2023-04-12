import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os
print(os.getcwd())  # Print current working directory
print(os.listdir())  # Print a list of files in the current working directory

app=Flask(__name__)
model=pickle.load(open('CKD.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/Prediction',methods=['POST','GET'])
def prediction():
    return render_template('indexnew.html')
@app.route('/Home',methods=['POST','GET'])
def my_home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features=[float(x) for x in request.form.values()]
    features_value=[np.array(input_features)]

    features_name=['blood_urea','blood glucose random','coronary_artery_disease','anemia','pus_cell','red_blood_cells','diabetesmellitus','pedal_edema']
    df=pd.DataFrame(features_value, columns=features_name)
    output=model.predict(df)
    return render_template('result.html',prediction_text=output)

if __name__=='__main__':
    app.run(debug=False)