from tkinter import *
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageTk
import pytesseract

# Set tesseract path if required (adjust path as needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Define the custom distillation_loss function
def distillation_loss(y_true, y_pred, temperature=3.0):
    soft_targets = tf.nn.softmax(y_true / temperature)
    student_output = tf.nn.softmax(y_pred / temperature)
    return tf.keras.losses.KLDivergence()(soft_targets, student_output) + \
           tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred) * 0.5

# Load the model with custom_objects
student_model_path = "C:\\Users\\msbha\\OneDrive\\Desktop\\Satellite Image Classification Using Hybrid Deep Learning Model\\GUI\\student_model.h5"
student_model = load_model(student_model_path, custom_objects={'distillation_loss': distillation_loss})
print("Model loaded successfully!")

# Define class labels
class_labels = {
    0: 'airport_runway', 1: 'artificial_grassland', 2: 'avenue', 3: 'bare_land', 4: 'bridge',
    5: 'city_avenue', 6: 'city_building', 7: 'city_green_tree', 8: 'city_road', 9: 'coastline',
    10: 'container', 11: 'crossroads', 12: 'dam', 13: 'desert', 14: 'dry_farm',
    15: 'forest', 16: 'fork_road', 17: 'grave', 18: 'green_farmland', 19: 'highway',
    20: 'hirst', 21: 'lakeshore', 22: 'mangrove', 23: 'marina', 24: 'mountain',
    25: 'mountain_road', 26: 'natural_grassland', 27: 'overpass', 28: 'parkinglot', 29: 'pipeline',
    30: 'rail', 31: 'residents', 32: 'river', 33: 'river_protection_forest', 34: 'sandbeach',
    35: 'sapling', 36: 'sea', 37: 'shrubwood', 38: 'snow_mountain', 39: 'sparse_forest',
    40: 'storage_room', 41: 'stream', 42: 'tower', 43: 'town', 44: 'turning_circle'
}

# GUI Application
root = Tk()
root.title("Satellite Image Classification")
root.geometry("600x550")
root.configure(bg="#f0f0f0")

# Title
Label(root, text="Satellite Image Classification", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333").pack(pady=15)

img_label = Label(root)
img_label.pack(pady=10)

label_result = Label(root, text="Prediction: ", font=('Helvetica', 14), bg="#f0f0f0")
label_result.pack(pady=10)

def contains_face(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def contains_text(img_cv):
    text = pytesseract.image_to_string(img_cv)
    return len(text.strip()) > 0

def classify_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Read and show image
            image = Image.open(file_path)
            image = image.resize((256, 256))
            tk_image = ImageTk.PhotoImage(image)
            img_label.configure(image=tk_image)
            img_label.image = tk_image

            # Preprocess for detection
            img_cv = cv2.imread(file_path)

            # Check for face or text
            if contains_face(img_cv):
                messagebox.showerror("Invalid Image", "Image contains a human face. Not a satellite image.")
                label_result.config(text="Prediction: Invalid - Contains Face")
                return

            if contains_text(img_cv):
                messagebox.showerror("Invalid Image", "Image contains text. Not a satellite image.")
                label_result.config(text="Prediction: Invalid - Contains Text")
                return

            # Preprocess for model
            img = cv2.resize(img_cv, (128, 128))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # Predict
            prediction = student_model.predict(img)
            confidence = np.max(prediction)
            class_index = np.argmax(prediction)
            predicted_label = class_labels.get(class_index, "Unknown")
            label_result.config(text=f"Prediction: {predicted_label}")

        except Exception as e:
            messagebox.showerror("Error", f"Could not process image.\n{str(e)}")

# Upload button
btn_upload = Button(root, text="Upload Image", command=classify_image, font=('Helvetica', 12), bg="#4caf50", fg="white", padx=10, pady=5)
btn_upload.pack(pady=10)

root.mainloop()

