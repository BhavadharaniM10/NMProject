from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import os

# Setup NLTK data path (for deployment compatibility)
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

for pkg in ("stopwords", "punkt", "punkt_tab"):
    nltk.download(pkg, download_dir=nltk_data_path)
    
# Safe downloads
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model = load_model('my_model.h5')  # Or 'lstm_model.h5' if you're using that
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocessing function
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return ' '.join(tokens)

# Prediction logic
def predict_text(text):
    if not text:
        return None, 'No text provided'

    processed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded_sequence = pad_sequences(sequence, maxlen=100, padding='post')  # Adjust maxlen if needed
    prediction = model.predict(padded_sequence, verbose=0)[0][0]
    result = 'Fake' if prediction < 0.5 else 'Real'
    confidence = float(prediction) if result == 'Real' else float(1 - prediction)
    return {'prediction': result, 'confidence': confidence}, None

# Home page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', result=None)

# Prediction from web form
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')
    result, error = predict_text(text)
    if error:
        return render_template('index.html', error=error)
    return render_template('index.html', result=result)

# API prediction (for POST JSON input)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '')
    result, error = predict_text(text)
    if error:
        return jsonify({'error': error}), 400
    return jsonify(result)

# Start the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
