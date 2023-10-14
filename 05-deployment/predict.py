from flask import Flask
import pickle
from flask import request
from flask import jsonify

model_file = 'model_C=1.0.bin'

# load model
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict(): 
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[:,1]
    churn = y_pred >= 0.5

    results = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 9696)