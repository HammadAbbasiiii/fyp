import re
import joblib
import numpy as np
from keras.utils import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
from keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load your models and other resources
# (Replace these with your actual model paths and imports)
port_stem = PorterStemmer()
tfidf_vect = joblib.load('tfidf_vectorizer.pkl')
lsvc = joblib.load('linear_svc_model.pkl')
lstm_model = load_model('lstm_model.h5')
ensemble_model = load_model('ensemble_model.h5')

with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_config = f.read()
    tokenizer = tokenizer_from_json(tokenizer_config)

with open('maxlen.txt', 'r') as f:
    maxlen = int(f.read())

# Function for stemming (You can use it when you uncomment the code above)
def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        data = request.json
        text_data = data['text']
        
        # Preprocess the new text data (You can uncomment this when you have your preprocessing code)
        text_data = stemming(text_data)
        text_data_tfidf = tfidf_vect.transform([text_data])
        
        # Predict using LinearSVC (Uncomment this when you have your model)
        lsvc_pred = lsvc.predict(text_data_tfidf)
        # print(f"lsvcValue is: {lsvc_pred}")
        # Tokenize and pad the input data for LSTM model (Uncomment this when you have your model)
        text_data_seq = tokenizer.texts_to_sequences([text_data])
        text_data_pad = pad_sequences(text_data_seq, padding='post', maxlen=maxlen)

        # Predict using LSTM model (Uncomment this when you have your model)
        lstm_pred = lstm_model.predict(text_data_pad)
        
        # Convert predictions to the correct data type (float32)
        lsvc_pred = lsvc_pred.astype('float32')

        
        lstm_pred = lstm_pred.astype('float32')

        

        # Reshape and stack the predictions horizontally (Uncomment this when you have your model)
        ensemble_input = np.hstack([lsvc_pred.reshape(-1, 1), lstm_pred.reshape(-1, 1)])

        

        # Predict using Ensemble model (Uncomment this when you have your model)
        ensemble_pred = ensemble_model.predict(ensemble_input)

        # print(f"ensemble value is: {ensemble_pred}")

        # Determine the result based on your threshold (Uncomment this when you have your model)
        prediction_value = ensemble_pred[0][0]
        
        

        # print(f"Prediction Value is: {prediction_value}")

        # Convert prediction to a JSON serializable format (float or int)
        prediction_value = float(ensemble_pred[0][0])  # Convert to float

        # Create a JSON response
        result = {"prediction": prediction_value}
        
        # Return the JSON response
        return jsonify(result)

        

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)





# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get the input data from the request
#         data = request.json
#         text_data = data['text']

#         # Preprocess the new text data
#         text_data = stemming(text_data)
#         text_data_tfidf = tfidf_vect.transform([text_data])

#         # Predict using LinearSVC
#         lsvc_pred = lsvc.predict(text_data_tfidf)

#         # Tokenize and pad the input data for LSTM model
#         text_data_seq = tokenizer.texts_to_sequences([text_data])
#         # text_data_pad = pad_sequences(text_data_seq, padding='post', maxlen=maxlen)

#         # Predict using LSTM model
#         # lstm_pred = lstm_model.predict(text_data_pad)

#         # Convert predictions to the correct data type (float32)
#         lsvc_pred = lsvc_pred.astype('float32')
#         lstm_pred = lstm_pred.astype('float32')

#         # Reshape and stack the predictions horizontally
#         ensemble_input = np.hstack([lsvc_pred.reshape(-1, 1), lstm_pred.reshape(-1, 1)])

#         # Predict using Ensemble model
#         ensemble_pred = ensemble_model.predict(ensemble_input)

#         # Determine the result based on your threshold
#         result = {"prediction": "fake" if ensemble_pred[0][0] >= 0.5 else "real"}
        
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)







# app = Flask(__name__)


# if __name__ == '__main__':
#     app.run(debug=True)

