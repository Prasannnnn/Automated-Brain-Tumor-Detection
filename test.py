from ultralytics import YOLO
import cv2
import os

# Load your trained model
model = YOLO('final_brain.pt')

# Folder with test images
image_folder = 'test_images'
output_folder = 'predicted_outputs'
os.makedirs(output_folder, exist_ok=True)

# Process each image
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        results = model.predict(source=image_path, conf=0.25)

        # Get names of detected classes in this image
        names = model.names  # class index to name mapping
        for r in results:
            print(f"\n✅ Image: {filename}")
            if r.boxes is not None and len(r.boxes.cls) > 0:
                for cls_idx in r.boxes.cls:
                    class_id = int(cls_idx)
                    class_name = names[class_id]
                    print(f"🔍 Detected: {class_name}")
            else:
                print("⚠️ No detections found.")

        # Save image with bounding boxes
        result_image = results[0].plot()
        output_path = os.path.join(output_folder, f'pred_{filename}')
        cv2.imwrite(output_path, result_image)
        print(f"🖼️ Saved annotated image: {output_path}")
