from flask import Flask, request, render_template
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
import joblib

app = Flask(__name__)

knn_7 = joblib.load('stacking/knn_7.joblib')
knn_8 = joblib.load('stacking/knn_8.joblib')
meta_clf = joblib.load('stacking/meta_clf.joblib')
accuracy = joblib.load('stacking/accuracy.joblib')

def stackingClassifier(data):
    result_n_7 = pd.DataFrame(knn_7.predict(data), columns=['P1'])
    result_n_8 = pd.DataFrame(knn_8.predict(data), columns=['P2'])
    combined_data = pd.concat([result_n_7, result_n_8], axis=1)

    prediction = meta_clf.predict(combined_data)
    score = accuracy 

    return prediction, score


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        data = {key: float(value) for key, value in data.items()}
        data = pd.DataFrame([data])
        prediction, score = stackingClassifier(data)
        return render_template('index.html', prediction=prediction[0], accuracy=score)
    except Exception as e:
        return render_template('index.html', error=str(e))
    
@app.route('/data')
def data_info():
    return render_template('data.html')


if __name__ == '__main__':
    app.run(debug=True)
