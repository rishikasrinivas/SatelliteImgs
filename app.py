import os
from flask import Flask, request, render_template
import pickle
import pandas as pd
from src.load_data import ProcessData
from PIL import Image
from torchvision import transforms
app = Flask("__name__")


@app.route("/", methods = ['get'])
def load():
    with app.app_context():
        file = "index.html"
        return render_template(file)
@app.route("/submit", methods = ['POST'])
def predict_lab():
    
    file = request.files['myimg']  # contains the input images from html

    # if the formated file is true and 'file' is not none
    if file:
        p=ProcessData()
        file_path = file.filename
        file_path = os.path.join('/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/Satellit/SatelliteImgs', file_path)
        file.save(file_path)  # saving the image into static folder

        convertToTensor = transforms.ToTensor()
        # the image is resize by 300x300
        img = Image.open(file_path)
        train_t, _=p.apply_transformations()
        ds=train_t(img)
        # convert the image into array, then divided by 255
    
        model=pickle.load(open("/Users/rishikasrinivas/Documents/Rishika/UCSC/Projects/Satellit/SatelliteImgs/model.sav", "rb"))
        # Stack arrays in sequence vertically
        

        # todo model predict the image
        classes = model(ds.unsqueeze(0))
        print("RESULTT", classes)
        if classes[0][0] > 0.5:
            label = "No Wildfire"
        else:
            label = "Wild"

        # the output will be display in 'predict.html' as label
        return render_template('index.html', prediction=label, user_image=file_path)

        

app.run(debug=True)