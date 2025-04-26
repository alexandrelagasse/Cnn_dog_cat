import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'src/model/cnn_dog_cat_improved.h5'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle une seule fois
model = load_model(MODEL_PATH)

# Fonction utilitaire
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    x = x / 255.0
    pred = model.predict(x)[0][0]
    label = 'Dog' if pred > 0.5 else 'Cat'
    confidence = float(pred) if pred > 0.5 else 1 - float(pred)
    return label, confidence

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    files = request.files.getlist('images')
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            label, confidence = predict_image(filepath)
            results.append({
                'filename': file.filename,
                'label': label,
                'confidence': f"{confidence*100:.2f}%",
                'filepath': filepath
            })
    return jsonify(results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True) 