import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import tempfile

# Load model and Haar cascade once
@st.cache_resource
def load_emotion_model():
    model = load_model("emotion_recognition_model.h5")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return model, face_cascade

model, face_cascade = load_emotion_model()
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Title
st.title("😄 Emotion Detection from Face Image")
st.write("Upload an image or take a picture to detect emotions.")

# Upload or Camera input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Or take a picture")

# Use whichever image is provided
image_data = uploaded_file if uploaded_file is not None else camera_image

if image_data is not None:
    # Convert to OpenCV format
    image = Image.open(image_data).convert("RGB")
    open_cv_image = np.array(image)
    frame = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_resized = np.expand_dims(face_resized, axis=-1)
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = face_resized / 255.0

        prediction = model.predict(face_resized)
        emotion_label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"{emotion_label} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Convert back to RGB and show image
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
else:
    st.info("Please upload an image or take a picture to proceed.")

