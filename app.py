import tensorflow.keras
from flask import Flask, render_template, request
import cv2
import numpy
import os
from keras.models import load_model
import base64
from keras.models import Sequential

app = Flask(__name__ ,static_url_path='/static')

print("Loading model....")
model = load_model('./models/my_model.h5',compile=False)
print("....Loaded model")


# Prediction
def traffic_prediction(test_img, model):
    classes = {0: 'Bien bao cam o to',
               1: 'Bien bao cam re trai',
               2: 'Bien bao cam re phai',
               3: 'Bien bao cam do xe',
               4: 'Bien bao duong khong bang phang',
               5: 'Bien bao huong phai di vong sang trai',
               6: 'Bien bao cam di nguoc chieu',
               7: 'Bien bao cam xe khach va xe tai',
               8: 'Bien bao toc do toi da cho phep',
               9: 'Bien bao cam dung xe va do xe',
               10: 'Bien bao het tat ca cac lenh cam',
               11: 'Bien bao cho ngoat nguy hiem',
               12: 'Bien bao duong giao nhau',
               13: 'Bien bao giao nhau duong khong uu tien',
               14: 'Bien bao giao nhau duong uu tien',
               15: 'Bien bao noi giao nhau co tin hieu den',
               16: 'Bien bao nguoi di bo cat ngang',
               17: 'Bien bao tre em',
               18: 'Bien nguoi di xe dap cat ngang',
               19: 'Bien bao nguy hiem khac',
               20: 'Bien bao noi giao nhau chay theo vong xuyen',
               21: 'Bien bao cho quay xe',
               22: 'Bien bao duong di bo',
               23: 'Bien bao benh vien',
               24: 'Bien bao tram xang',
               25: 'Bien bao phan lan'
               }
    image = cv2.resize(test_img, (30, 30), interpolation=cv2.INTER_AREA)
    prediction = model.predict(image.reshape(-1, 30, 30))
    predict_img = numpy.argmax(prediction, axis=-1)
    pred_name = classes.get(predict_img[0])

    return pred_name


@app.route("/")
def home():
    """
    This function renders the html page. This is the root(entrypoint) for the flask app.

    """

    return render_template("index.html")


@app.route('/api/v1', methods=['GET','POST'])
def traffic_light_prediction():
    """
        This is the API for traffic sign prediction.

        The image is uploaded by the user for classification. The image is saved on the server. The prediction function is called. The time taken to predict is printed as log trace.
        The output (predicted sign) is rendered as a page in the flask app.
    """
    if request.method == 'POST':
        img = request.files.getlist('image')
        img = cv2.imdecode(numpy.fromstring(img, numpy.uint8), 1)
        # preds = traffic_prediction(img, model)

        print(img)


    # return render_template('index.html', reply =preds)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8050)