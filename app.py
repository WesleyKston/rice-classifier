from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)

# ✅ Google Drive direct link for the .h5 model
file_id = '1Vgtrw1Lf7KfLO-sbB8Iytfiyz9J5DTRR'
model_url = f'https://drive.google.com/uc?id=1Vgtrw1Lf7KfLO-sbB8Iytfiyz9J5DTRR'
model_path = 'rice_type_classification_with_fluid_mechanics.h5'

# ✅ Download the model if not already present
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# ✅ Load the model
model = tf.keras.models.load_model(model_path)

# ✅ Class labels mapping
class_mapping = {
    0: 'arborio',
    1: 'basmati',
    2: 'ipsala',
    3: 'jasmine',
    4: 'karacadag'
}

# ✅ Image preprocessing function
def preprocess_image(image_data):
    img = Image.open(image_data).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        fluid_behavior_str = request.form['fluid_behavior']

        if not image:
            return render_template('index.html', error="Please upload an image.")

        # ✅ Map fluid behavior string to int
        behavior_map = {
            "Sink": 0,
            "Float": 1
        }
        fluid_behavior = behavior_map.get(fluid_behavior_str, 0)

        try:
            input_image = preprocess_image(io.BytesIO(image.read()))
            fluid_input = np.array([[fluid_behavior]], dtype="float32")

            # ✅ Predict
            prediction = model.predict([input_image, fluid_input])
            predicted_class = np.argmax(prediction, axis=1)[0]
            predicted_label = class_mapping[predicted_class]

            return render_template('index.html', prediction=predicted_label)

        except Exception as e:
            print("Prediction error:", e)
            return render_template('index.html', error="Prediction failed. Check server logs.")

    return render_template('index.html')

# ✅ Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
