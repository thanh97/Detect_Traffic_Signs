from flask import Flask, render_template, request, send_from_directory
import cv2

import numpy as np
from skimage import io, color, exposure, transform
import os
from keras import backend as K

K.set_image_data_format('channels_first')
from keras.models import load_model

from datetime import datetime

import joblib

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

NUM_CLASSES = 30
IMG_SIZE = 30

KERAS_MODEL = './models/my_model.h5'

print("Loading model....")
model = load_model(KERAS_MODEL)
print("....Loaded model")

CLASSIFICATION_LABEL = "models/classification_labels.pkl"

classification_labels = joblib.load(CLASSIFICATION_LABEL)
classification_labels = list(classification_labels)


def preprocess_img(img):
    """
    This function pre-processes the image that is given for classification. This is the same pre-processing used for input images during model training.

    The images are normalized for histogram and rescaled to standard size.

    INPUT: 
        img: an input image for classification. The expected input is in png format.

    OUTPUT:
        img: the input image after pre-processing

    """
    # Extract img from object
    if img.dtype == 'object':
        img = img[0]

    # Convert RGBA to rgB
    if img.shape[2] == 4:
        img = color.rgba2rgb(img)

    # Resize img
    ratio = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * ratio))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0] // 2, img.shape[1] // 2
    img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
          centre[1] - min_side // 2:centre[1] + min_side // 2,
          :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, 1)

    return img


def get_class(img_path):
    """
    This function returns the class of the input image. 

    Each image is stored within a folder that is named after its class. 


    INPUT: 
        img_path: path of the current image

    OUTPUT:
        The class of the image

    """

    return int(img_path.split('/')[-2])


# Prediction
def traffic_prediction(test_img):
    """
    This function returns the predicted class of the image.

    INPUT: 
        test_img: The image that has to be classified 

    OUTPUT:
        The class/prediction of the image

    """

    X_test = []
    X_test.append(preprocess_img(io.imread(test_img)))
    X_test = np.array(X_test)
    y_pred = model.predict_classes(X_test)
    return y_pred


@app.route("/")
def home():
    """
    This function renders the html page. This is the root(entrypoint) for the flask app.

    """

    return render_template("index.html")


@app.route('/api/v1', methods=['GET', 'POST'])
def traffic_light_prediction():
    """
        This is the API for traffic sign prediction.

        The image is uploaded by the user for classification. The image is saved on the server. The prediction function is called. The time taken to predict is printed as log trace. 
        The output (predicted sign) is rendered as a page in the flask app.
    """

    t = datetime.now()
    # query = request.form['pic']
    # imgFile = request.form['pic']

    # print("Saving the file")

    # save the image
    imgFile = request.files['pic']
    f = os.path.join(app.config['UPLOAD_FOLDER'], imgFile.filename)

    imgFile.save(f)
    # print("file saved: ", f)
    # print("uploaded image saved")

    response_num = traffic_prediction(f)
    response_label = classification_labels[int(response_num)]

    dt = datetime.now() - t

    print('Prediction Time (hh:mm:ss.ms) {}'.format(dt))
    print("Image %s classified as %s" % (f, response_label))

    # reply = "Welcome to this awesome page"
    # return render_template('index.html', query=query, reply=reply)
    return render_template('index.html', inputTrafficSign="/" + f,
                           reply=response_label)


@app.route("/uploads/<filename>")
def send_file(filename):
    """
        This function uploads the file from local/host machine to the server. 
    """

    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/get/<string:query>")
def get_raw_response(query):
    """
        This is a placeholder function to test the app. 
    """

    return str((query))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
