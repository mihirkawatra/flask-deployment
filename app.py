import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# load the model from disk
MODEL_DIR = './models/'
salary_model = pickle.load(open(MODEL_DIR+'salary_model.pkl', 'rb'))
spam_model = pickle.load(open(MODEL_DIR+'spam_model.pkl', 'rb'))
spam_cv = pickle.load(open(MODEL_DIR+'spam_transform.pkl', 'rb'))
sentiment_cv = pickle.load(open(MODEL_DIR+'sentiment_transform.pkl', 'rb'))
sentiment_model = pickle.load(open(MODEL_DIR+'sentiment_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/email')
def email():
    return render_template('home.html')

@app.route('/salary', methods=['POST'])
def predict_salary():
    '''
    For rendering results on HTML GUI
    '''
    print(request.form)
    format = request.args.get('format')

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = salary_model.predict(final_features)

    output = round(prediction[0], 2)

    if(format == 'json'):
        return jsonify({'salary': output})

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


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
        return render_template('result.html', prediction=my_prediction)
    else:
        return render_template('result.html', prediction=-1)

@app.route('/sentiment', methods=['POST'])
def predict_sentiment():
    format = request.args.get('format')
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = sentiment_cv.transform(data).toarray()
        my_prediction = sentiment_model.predict(vect)
        if(format == 'json'):
            pred = 'Positive' if my_prediction == 1 else 'Negative'
            return jsonify({'prediction': pred})
        return render_template('result.html', prediction=my_prediction)
    else:
        return render_template('result.html', prediction=-1)

if __name__ == "__main__":
    app.run(debug=True)  # auto-reload on code change
