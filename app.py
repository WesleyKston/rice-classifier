from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)

model_url = 'https://drive.google.com/file/d/1PaZdjYFcjVSX-qTbC2TzIV-I5wRwU0gu/view?usp=sharing'
model_path = 'rice_type_classification_model.keras'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)
    
model = tf.keras.models.load_model(model_path)

class_mapping = {0: 'arborio', 1: 'basmati', 2: 'ipsala', 3: 'jasmine', 4: 'karacadag'}

def preprocess_image(image_data):
    img = Image.open(image_data).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        fluid_behavior = int(request.form['fluid_behavior'])

        if not image:
            return render_template('index.html', error="Please upload an image.")

        input_image = preprocess_image(io.BytesIO(image.read()))
        fluid_input = np.array([[fluid_behavior]]).astype("float32")

        # ✅ Simple prediction call now works
        prediction = model.predict([input_image, fluid_input])
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_label = class_mapping[predicted_class]

        return render_template('index.html', prediction=predicted_label)

    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

