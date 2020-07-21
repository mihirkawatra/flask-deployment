import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
import urllib.request
import pickle
from keras_preprocessing import image
from keras.models import load_model
from keras.utils.data_utils import get_file
import numpy as np
import tensorflow as tf
import os
import re

app = Flask(__name__)

# load the model from disk
MODEL_DIR = './models/'
STATIC_FOLDER = 'static'
# Path to the folder where we'll store the upload before prediction
UPLOAD_FOLDER = STATIC_FOLDER + '/uploads'

print('[INFO] : Model loading ................')
model = load_model(MODEL_DIR + 'cat_dog_classifier.h5')
graph = tf.get_default_graph()
salary_model = pickle.load(open(MODEL_DIR+'salary_model.pkl', 'rb'))
spam_model = pickle.load(open(MODEL_DIR+'spam_model.pkl', 'rb'))
spam_cv = pickle.load(open(MODEL_DIR+'spam_transform.pkl', 'rb'))
sentiment_pipeline = pickle.load(open(MODEL_DIR+'sentiment_pipeline.pkl', 'rb'))
iris_model = pickle.load(open(MODEL_DIR+'xgboost_iris.pkl', 'rb'))
print('[INFO] : Models loaded')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/email')
def email():
    return render_template('spam.html')

@app.route('/salary', methods=['GET', 'POST'])
def predict_salary():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        print(request.form)
        format = request.args.get('format')

        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = salary_model.predict(final_features)

        output = round(prediction[0], 2)

        if(format == 'json'):
            return jsonify({'salary': output})

        return render_template('salary.html', prediction_text='Employee Salary should be $ {}'.format(output))
    else:
        return render_template('salary.html')

@app.route('/iris', methods=['GET', 'POST'])
def predict_iris():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        print(request.form)
        format = request.args.get('format')

        int_features = list(map(float, [request.form['sepal length'], request.form['sepal width'], request.form['petal length'], request.form['petal width']]))
        final_features = np.array(int_features).reshape(1, -1)
        prediction = iris_model.predict(final_features)

        output = round(prediction[0], 2)
        target_names = ['Setosa', 'Versicolor', 'Virginica']
        if(format == 'json'):
            return jsonify({'iris': target_names[output]})

        return render_template('iris.html', prediction_text='Prediction: Iris {}'.format(target_names[output]))
    else:
        return render_template('iris.html')

@app.route('/email', methods=['POST'])
def predict_email():
    format = request.args.get('format')
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = spam_cv.transform(data).toarray()
        my_prediction = spam_model.predict(vect)
        if(format == 'json'):
            pred = 'Spam' if my_prediction == 1 else 'Not Spam'
            return jsonify({'prediction': pred})
        return render_template('spam_result.html', prediction=my_prediction)
    else:
        return render_template('result.html', prediction=-1)

@app.route('/sentiment', methods=['GET', 'POST'])
def predict_sentiment():
    format = request.args.get('format')
    if request.method == 'POST':
        message = request.form['message']
        TAG_RE = re.compile(r'<[^>]+>')
        TAG_RE.sub('', message)
        data = [message]
        my_prediction = sentiment_pipeline.predict(data)
        if(format == 'json'):
            pred = 'Positive' if my_prediction == 1 else 'Negative'
            return jsonify({'prediction': pred})
        return render_template('sentiment_result.html', prediction=my_prediction)
    else:
        return render_template('sentiment.html')

def img_predict(fullpath):
    data = image.load_img(fullpath, target_size=(128, 128, 3))
    # (150,150,3) ==> (1,150,150,3)
    data = np.expand_dims(data, axis=0)
    # Scaling
    data = data.astype('float') / 255

    # Prediction

    with graph.as_default():
        result = model.predict(data)

    return result

# Process file and predict his label
@app.route('/dogcat', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('dog.html')
    else:
        file = request.files['image']
        fullname = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(fullname)

        result = img_predict(fullname)

        pred_prob = result.item()

        if pred_prob > .5:
            label = 'Dog'
            accuracy = round(pred_prob * 100, 2)
        else:
            label = 'Cat'
            accuracy = round((1 - pred_prob) * 100, 2)

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)

@app.route('/dogcat/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)  # auto-reload on code change
