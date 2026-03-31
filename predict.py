import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import config

# Load model
model = tf.keras.models.load_model(config.MODEL_PATH)


def predict_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    
    predicted_class = config.class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence