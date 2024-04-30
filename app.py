from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import json

app = Flask(__name__)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = 'artifact_downloads/outputs/model.pkl'


try:
    logger.info("Loading model from path: %s", model_path)
    model = joblib.load(model_path)
    logger.info("Model loading successful.")
except Exception as e:
    logger.error("An error occurred while loading the model: %s", str(e))
    raise RuntimeError("Failed to load the model")

def classification_result(data, model):
    try:
        # Prepare data for prediction
        df = pd.DataFrame([data])

        # Predict
        result = model.predict(df)

        if isinstance(result, pd.DataFrame):
            result = result.values

        return {'Results':result.tolist()}

    except Exception as e:
        logger.exception("An error occurred during prediction: %s", str(e))
        raise RuntimeError("Failed to make prediction")


@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Log received data
        logger.info('Received data: %s', data)

        # Preprocess the data and make prediction
        result = classification_result(data, model)

        output = {'Results': result}

        # Log response
        logger.info('Prediction: %s', output)

        return jsonify(output), 200
    
    except Exception as e:
        logger.exception('An error occurred: %s', e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')



# Example CURL command to invoke the API
# curl -X POST http://localhost:5000/prediction -H 'Content-Type: application/json' -d '{"PatientID": 1020531, "Pregnancies": 3, "PlasmaGlucose": 125, "DiastolicBloodPressure": 82, "TricepsThickness": 23, "SerumInsulin": 112, "BMI": 34.95472243, "DiabetesPedigree": 0.204847272, "Age": 46}'