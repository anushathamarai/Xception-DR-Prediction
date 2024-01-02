import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import requests
from flask import Flask, request, render_template, redirect, url_for
from cloudant.client import Cloudant

app=Flask(__name__)

model=load_model(r"updated-xception-diabetic-retinopathy.h5")

# a simple page that says hello
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predict', methods=["GET", "POST"])
def res():
	if request.method=="POST":
		f=request.files['image']
		basepath=os.path.dirname(__file__)
		#print("Current path :", basepath)
		filepath=os.path.join(basepath,'uploads',f.filename)
		print("File: ", filepath)
		f.save(filepath)

		img=image.load_img(filepath, target_size=(299,299))
		x=image.img_to_array(img)
		x=np.expand_dims(x, axis=0)
		img_data=preprocess_input(x)
		prediction=np.argmax(model.predict(img_data), axis=1)

		index=['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'proliferative DR']
		result=str(index[prediction[0]])
		print(result)

		return render_template("prediction.html", prediction=result)
