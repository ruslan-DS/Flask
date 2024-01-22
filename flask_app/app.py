from flask import Flask, render_template, request
from model_yolo.model import load_model, load_transform
import numpy as np
from PIL import Image

app = Flask('My app')

model, transform = None, None


@app.before_request
def load_all():
    global model
    global transform

    model = load_model()
    transform = load_transform()


@app.route('/')
def general_page():
    return render_template('index.html')


@app.post('/predict')
def predict():

    if 'image' in request.files:

        image = np.array(Image.open(request.files['image']))

        transformed_image = transform(image)

        predict_image = model.predict(transformed_image)[0].plot()

        Image.fromarray(predict_image).show()

        # return render_template('result.html', Image.fromarray(predict_image)) # сделать так, чтобы сдетектированное изображение отображалось на html-странице

    else:
        print('Запрос пустой')


if __name__ == '__main__':
    app.run(debug=True)