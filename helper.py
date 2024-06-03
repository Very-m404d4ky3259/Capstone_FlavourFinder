import tensorflow as tf
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import Model

# Function to load class names from a file
def load_class_names(filepath):
    with open(filepath, 'r') as file:
        class_names = file.read().splitlines()
    return class_names

# Path to the class names file
class_names_Path = './Assets/class.txt'

# Load class names from the file
class_names = load_class_names(class_names_Path)  # Replace with your actual class names

model = tf.keras.models.load_model('./Assets/model.h5')

def preprocess_image(base64_img, target_size):
    # Decode the base64 image
    img_data = base64.b64decode(base64_img)
    img = Image.open(BytesIO(img_data))
    # Calculate the new size while maintaining the aspect ratio
    img.thumbnail(target_size, Image.LANCZOS)
    # Create a new image with white background
    new_img = Image.new("RGB", target_size, (255, 255, 255))
    # Paste the resized image onto the white background
    new_img.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))
    # Convert the image to a numpy array
    img_array = image.img_to_array(new_img)
    # Expand dimensions to match the shape the model expects
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image (this depends on how your model was trained, e.g., rescale to [0, 1] or [-1, 1])
    img_array = img_array / 255.0
    return img_array

# Function to predict the image class
def predict_image(model, img_array):
    # Get predictions
    predictions = model.predict(img_array)
    return predictions

def classification(payload, target_size=(224, 224), threshold=0.95):
    try:
        img_array = preprocess_image(payload, target_size)
    except Exception as e:
        return "Failed to process image: " + "payload not valid", 0
    
    predictions = predict_image(model, img_array)
    class_index = np.argmax(predictions[0])
    percentage = predictions[0][class_index]
    class_name = class_names[class_index]

    if percentage > threshold:
        return "Detected as: " + class_name, percentage * 100
    else:
        return "Not detected", 0
