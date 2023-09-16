# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:28:25 2023

@author: ACER
"""
from flask import Flask , request, render_template
def upload():
    if request.method == 'POST':
        f = request.files['url']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128)) 
        x = image.img_to_array(img)
        print(x)
        x = np.expand_dims(x,axis =0)
        print(x)
        y=model.predict(x)
        preds=np.argmax(y,axis=1)
       # preds = model.predict_classes(x)
        print("prediction",preds)
        index = ['Breast Cancer Negative','Breast Cancer Positive']
        text = str(index[preds[0]])
    return text


userEmail = request.form['userEmail']
userPassword = request.form['userPassword']
return userEmail, userPassword