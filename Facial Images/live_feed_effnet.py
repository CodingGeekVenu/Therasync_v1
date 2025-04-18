import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load trained EfficientNet B5 model
model = load_model("C:/Users/venum/Downloads/Therasync 1lasttry/autism-B3_87.0.h5")

# Check model input shape
print("Model expects input shape:", model.input_shape)  # Debugging

# Initialize webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to model's expected input size (200x200)
    resized = cv2.resize(frame, (200, 200))

    # Convert to numpy array and preprocess
    input_data = np.expand_dims(resized, axis=0)
    input_data = preprocess_input(input_data)  # Normalize using EfficientNet preprocessing

    # Predict ASD vs. Non-ASD
    prediction = model.predict(input_data)

    # Extract the first value from the array
    predicted_value = float(prediction[0][0])  # Convert to float
    print(f"Raw Prediction Value: {predicted_value}")  # Debugging output


    # Assign label based on threshold
    class_label = "ASD Detected" if predicted_value > 0.90 else "No ASD"

    # Display prediction on the live video
    cv2.putText(frame, f"Prediction: {class_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live ASD Detection - EfficientNet B5", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
