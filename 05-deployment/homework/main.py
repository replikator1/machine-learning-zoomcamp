from flask import Flask
import pickle
from flask import request
from flask import jsonify



app = Flask('credit')

#client = {"job": "retired", "duration": 445, "poutcome": "success"}

# Load model and dicvec from pickle files
with open('dv.bin', 'rb') as file1:
    dv = pickle.load(file1)
with open('model2.bin', 'rb') as file2:
    model = pickle.load(file2)

# Predict probability for var client
@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform(client)
    y_pro = model.predict_proba(X)[:,1][0]
    results = {"credit_probability":y_pro}

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 9696)