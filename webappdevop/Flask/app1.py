
import numpy as np
import os
from tensorflow.keras.models import load_model
from flask import Flask , request, render_template
from feature import FeatureExtraction
#from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer


app = Flask(__name__)
model = load_model("phishing_model.h5", compile=False)
model.compile()

                 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods = ['GET','POST'])
def form_post():
  if request.method == 'POST':
    url = request.form['url']
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1,30)
    
    #y='Normal URL'
    y_pred =model.predict(x)[0]
    
   
    if(y_pred >= 0.48):
        text ="Normal URL"
    else:
        text ="This URL is something suspicious"
        
   #1 is safe       
   #-1 is unsafe
    #y_pro_phishing = model.predict_proba(x)[0,0]
    #y_pro_non_phishing = model.predict_proba(x)[0,1]
   # if(y_pred ==1 ):
    #pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
    #return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
    
    return text
  return render_template('index.html')
  
if __name__ == '__main__':
    app.run(debug = False, threaded = False,port=9000)
