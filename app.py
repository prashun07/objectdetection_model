import sys
import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response
from werkzeug.exceptions import BadRequest
import os

# creating flask app
app = Flask(__name__)

model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)

def get_prediction(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model(img, size=640)
    return results
# get method
@app.route('/', methods=['GET'])
def get():
    return render_template("index.html")


# post method
@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read()

    # get prediction from model
    results = get_prediction(
        img_bytes, model)
    

    # updates results.imgs with boxes and labels
    results.render()

    # encoding the resulting image and return it
    for img in results.imgs:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', RGB_img)[1]
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
    return response


def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")

    file = request.files['file']

    if file.filename == '':
        raise BadRequest("Given file is invalid")

    return file


if __name__ == '__main__':
    # starting app
    app.run(debug=True, host='0.0.0.0')
