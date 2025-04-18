import cv2
import numpy as np
from keras.models import load_model

# ✅ Load the trained model
model = load_model('ASD_model.h5')  # Ensure the file is in the same directory

# ✅ Open the webcam
cap = cv2.VideoCapture(1)  # 0 for default webcam
IMG_SIZE = 200  # Must match the model input size

while True:
    ret, frame = cap.read()  # Capture frame
    if not ret:
        break

    # ✅ Convert to grayscale (if needed)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✅ Resize to match the model input size
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)) / 255.0  # Normalize

    # ✅ Reshape to match the model input format
    reshaped = resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # ✅ Make a prediction
    prediction = model.predict(reshaped)[0][0]

    # ✅ Determine class label
    label = "ASD Detected" if prediction > 0.5 else "Non-ASD"

    # ✅ Display the result
    cv2.putText(frame, f"Prediction: {label} ({prediction:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Live ASD Detection", frame)

    # ✅ Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
