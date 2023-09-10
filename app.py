from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model_name = "breast_logic.pkl"
with open(model_name, 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data or 'data' not in input_data:
            return jsonify({'error': 'Invalid input format'})
        input_features = input_data['data']
        input_features = np.asarray(input_features).reshape(1, -1)
        prediction = model.predict(input_features)
        if prediction[0] == 1:
            result = "Malignant"
        else:
            result = "Benign"

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
