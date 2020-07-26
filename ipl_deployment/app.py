import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle 


#Init Flask 
app = Flask(__name__)

 

model =  pickle.load(open('finalized_model.sav', 'rb'))


#To launch home page
@app.route('/')
def home():
    #Loading a HTML Page
    return render_template('index.html')




#Prediction Page
@app.route('/y_predict',methods=['POST'])
def y_predict():
    
    #Get all the user entered values from the form
    x_test = [[int(x) for x in request.form.values()]]
    
    x_test = np.array(x_test).reshape((1, -1))
   
   

    prediction = model.predict(x_test)
   
   
   
    return render_template('index.html',prediction_text=prediction)
if __name__ == "__main__":
    app.run(debug=True)
