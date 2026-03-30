import tensorflow as tf
import numpy as np
import sys

def run_inference(image_path, model_path="resnet50_model_2blocks.h5"):
    # Load the fine-tuned model you saved in your notebook
    model = tf.keras.models.load_model(model_path)

    # Preprocessing logic from your research
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    print(f"Inference complete. Raw predictions: {predictions}")
    return predictions

if __name__ == "__main__":
    print("Medical Image Modality Identification System")