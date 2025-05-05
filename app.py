import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Load the model once
model = YOLO("final_brain.pt")
class_names = model.names  # class index to name

st.title("🧠 AUTOMATED BRAIN TUMOR DETECTION")
st.write("Upload a brain scan image to detect tumor types using YOLOv8.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Predict using the YOLO model
    results = model.predict(source=temp_path, conf=0.25)

    # Process results
    r = results[0]
    img_with_boxes = r.plot()  # get image with bounding boxes
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)

    # Display the image
    st.image(img_with_boxes, caption="Detected Image", use_column_width=True)

    # Display detected classes
    if r.boxes is not None and len(r.boxes.cls) > 0:
        st.subheader("Detected Tumor Types:")
        for cls_idx in r.boxes.cls:
            class_id = int(cls_idx)
            class_name = class_names[class_id]
            st.success(f"✅ {class_name}")
    else:
        st.warning("⚠️ No tumor detected.")

    # Clean up temp file
    os.remove(temp_path)
