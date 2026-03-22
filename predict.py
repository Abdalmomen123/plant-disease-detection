import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import config

# Load model
model = tf.keras.models.load_model(config.MODEL_PATH)


def predict_image(img_path):
    img = image.load_img(img_path, target_size=config.IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = config.class_names[np.argmax(predictions)]

    print("Prediction:", predicted_class)