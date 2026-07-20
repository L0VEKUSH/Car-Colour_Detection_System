import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input


class CarColorDetector:

    def __init__(self, model_path="models/car_color_model.keras"):

        self.model = tf.keras.models.load_model(model_path)

        self.class_names = [
            "beige",
            "black",
            "blue",
            "brown",
            "gold",
            "green",
            "grey",
            "orange",
            "pink",
            "purple",
            "red",
            "silver",
            "tan",
            "white",
            "yellow",
        ]

    def detect_dominant_color(self, crop):

        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        try:
            image = cv2.resize(crop, (224, 224))

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = image.astype(np.float32)

            # Match EfficientNet preprocessing
            image = preprocess_input(image)

            image = np.expand_dims(image, axis=0)

            prediction = self.model.predict(image, verbose=0)[0]

            index = np.argmax(prediction)

            confidence = float(prediction[index]) * 100

            return self.class_names[index].title(), round(confidence, 2)

        except Exception:
            return "Unknown", 0.0

    @staticmethod
    def is_blue(color_name):
        return color_name.lower() == "blue"