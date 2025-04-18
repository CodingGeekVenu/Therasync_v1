import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# ✅ Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# ✅ Capture Webcam Video
cap = cv2.VideoCapture(0)

# ✅ Store Fixation Data
eye_x, eye_y = [], []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in [474, 475, 476, 477]:  # Landmark indexes for eyes
                x = int(face_landmarks.landmark[landmark].x * frame.shape[1])
                y = int(face_landmarks.landmark[landmark].y * frame.shape[0])

                # ✅ Store X, Y Eye Position
                eye_x.append(x)
                eye_y.append(y)

                # ✅ Draw Fixation Point
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Webcam Eye Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# ✅ Step 3: Generate Scanpath from Webcam Tracking
plt.figure(figsize=(10, 6), facecolor='black')
plt.scatter(eye_x, eye_y, c=np.linspace(0, 1, len(eye_x)), cmap="cool", alpha=0.7, label="Fixations")
plt.plot(eye_x, eye_y, color='orange', linestyle='-', linewidth=1.5, alpha=0.8, label="Saccades")

plt.gca().invert_yaxis()  # Flip Y-axis for eye-tracking format
plt.axis("off")
plt.legend()
plt.title("Webcam Eye-Tracking Scanpath", color='white')
plt.show()
