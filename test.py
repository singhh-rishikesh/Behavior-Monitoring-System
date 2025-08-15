# # from flask import Flask, render_template, Response
# # import cv2
# # import mediapipe as mp
# # from deepface import DeepFace
# # import numpy as np

# # app = Flask(__name__)

# # # Initialize AI models
# # mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
# # mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# # mp_draw = mp.solutions.drawing_utils

# # # Open webcam
# # camera = cv2.VideoCapture(0)

# # def detect_eye_direction(landmarks):
# #     """ Estimates eye movement direction using facial landmarks. """
# #     left_eye_x = (landmarks[33][0] + landmarks[133][0]) // 2
# #     right_eye_x = (landmarks[362][0] + landmarks[263][0]) // 2
# #     nose_x = landmarks[1][0]

# #     left_eye_y = (landmarks[33][1] + landmarks[133][1]) // 2
# #     right_eye_y = (landmarks[362][1] + landmarks[263][1]) // 2
# #     nose_y = landmarks[1][1]

# #     if left_eye_x > nose_x and right_eye_x > nose_x:
# #         return "Looking Left 👀"
# #     elif left_eye_x < nose_x and right_eye_x < nose_x:
# #         return "Looking Right 👀"
# #     elif left_eye_y < nose_y and right_eye_y < nose_y:
# #         return "Looking Up ⬆️"
# #     elif left_eye_y > nose_y and right_eye_y > nose_y:
# #         return "Looking Down ⬇️"
# #     else:
# #         return "Looking Center 👀"

# # def classify_hand_gesture(landmarks):
# #     """ Classifies various hand gestures based on finger positions. """
# #     thumb_up = landmarks[4][1] < landmarks[3][1]  # Thumb above base
# #     index_up = landmarks[8][1] < landmarks[6][1]  # Index finger up
# #     middle_up = landmarks[12][1] < landmarks[10][1]
# #     ring_up = landmarks[16][1] < landmarks[14][1]
# #     pinky_up = landmarks[20][1] < landmarks[18][1]

# #     if thumb_up and not (index_up or middle_up or ring_up or pinky_up):
# #         return "Thumbs Up 👍"
# #     elif all([index_up, middle_up, ring_up, pinky_up]):
# #         return "Open Palm 🖐️"
# #     elif not any([index_up, middle_up, ring_up, pinky_up, thumb_up]):
# #         return "Fist ✊"
# #     elif index_up and not (middle_up or ring_up or pinky_up):
# #         return "Pointing ☝️"
# #     elif index_up and middle_up and not (ring_up or pinky_up):
# #         return "Peace Sign ✌️"
# #     elif thumb_up and pinky_up and not (index_up or middle_up or ring_up):
# #         return "Call Me 🤙"
# #     elif index_up and pinky_up and not (middle_up or ring_up):
# #         return "Rock Sign 🤘"
# #     elif all([index_up, middle_up, ring_up]) and not pinky_up:
# #         return "Stop ✋"
# #     else:
# #         return "Unknown Gesture"

# # def generate_frames():
# #     while True:
# #         success, frame = camera.read()
# #         if not success:
# #             break
# #         else:
# #             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #             # Face Mesh for Eye Tracking
# #             face_results = mp_face_mesh.process(frame_rgb)
# #             eye_status = "No Face Detected"

# #             if face_results.multi_face_landmarks:
# #                 for face_landmarks in face_results.multi_face_landmarks:
# #                     landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]
# #                     eye_status = detect_eye_direction(landmarks)

# #             # Emotion Detection
# #             emotion_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
# #             emotion = emotion_result[0]['dominant_emotion']

# #             # Hand Gesture Recognition
# #             hand_results = mp_hands.process(frame_rgb)
# #             gesture_text = "No Hand Detected"

# #             if hand_results.multi_hand_landmarks:
# #                 for hand_landmarks in hand_results.multi_hand_landmarks:
# #                     mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                    
# #                     # Convert landmark points
# #                     landmarks = []
# #                     for landmark in hand_landmarks.landmark:
# #                         h, w, _ = frame.shape
# #                         landmarks.append((int(landmark.x * w), int(landmark.y * h)))
                    
# #                     # Classify the gesture
# #                     gesture_text = classify_hand_gesture(landmarks)

# #             # Display results
# #             cv2.putText(frame, f"Eye: {eye_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# #             cv2.putText(frame, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# #             cv2.putText(frame, f"Hand Gesture: {gesture_text}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

# #             _, buffer = cv2.imencode('.jpg', frame)
# #             frame = buffer.tobytes()
# #             yield (b'--frame\r\n'
# #                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/video')
# # def video():
# #     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # if __name__ == "__main__":
# #     app.run(debug=True)


# from flask import Flask, render_template, Response
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import csv
# import os
# import numpy as np

# app = Flask(__name__)

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Open webcam
# camera = cv2.VideoCapture(0)

# # Ensure CSV file exists
# CSV_FILE = "detection_data.csv"
# if not os.path.exists(CSV_FILE):
#     with open(CSV_FILE, 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Eye Movement", "Emotion", "Gesture", "Pointing At"])

# # Function to log data
# def log_data(eye_movement, emotion, gesture, pointed_object):
#     with open(CSV_FILE, 'a', newline='', encoding="utf-8") as file:
#         writer = csv.writer(file)
#         writer.writerow([eye_movement, emotion, gesture, pointed_object])

# # Object Mapping Based on Direction
# direction_to_object = {
#     "Left": "💊 Medicine kept on the table",
#     "Right": "💧 Water bottle is there",
#     "Up": "🌀 Fan should be turned on",
#     "Down": "⬇️ Something is on the floor",
#     "Center": "Turn on the TV."
# }

# # Function to detect eye movement
# def detect_eye_movement(landmarks):
#     left_eye_x = (landmarks[33].x + landmarks[133].x) / 2
#     right_eye_x = (landmarks[362].x + landmarks[263].x) / 2
#     nose_x = landmarks[1].x

#     left_eye_y = (landmarks[33].y + landmarks[133].y) / 2
#     right_eye_y = (landmarks[362].y + landmarks[263].y) / 2
#     nose_y = landmarks[1].y

#     # Debugging Info
#     print(f"Left Eye X: {left_eye_x}, Right Eye X: {right_eye_x}, Nose X: {nose_x}")
#     print(f"Left Eye Y: {left_eye_y}, Right Eye Y: {right_eye_y}, Nose Y: {nose_y}")

#     if left_eye_x > nose_x and right_eye_x > nose_x + 0.03:  # Added threshold
#         return "Left"
#     elif left_eye_x < nose_x and right_eye_x < nose_x - 0.03:  # Added threshold
#         return "Right"
#     elif left_eye_y < nose_y - 0.02 and right_eye_y < nose_y - 0.02:  # Up movement
#         return "Up"
#     elif left_eye_y > nose_y + 0.02 and right_eye_y > nose_y + 0.02:  # Down movement
#         return "Down"
#     else:
#         return "Center"

# # Function to detect hand gestures
# def detect_hand_gesture(landmarks):
#     index_up = landmarks[8].y < landmarks[6].y
#     middle_up = landmarks[12].y < landmarks[10].y
#     ring_up = landmarks[16].y < landmarks[14].y
#     pinky_up = landmarks[20].y < landmarks[18].y
#     thumb_up = landmarks[4].y < landmarks[3].y

#     if index_up and not (middle_up or ring_up or pinky_up):
#         return "Pointing ☝️"
#     elif all([index_up, middle_up, ring_up, pinky_up]):
#         return "Open Palm 🖐️"
#     elif not any([index_up, middle_up, ring_up, pinky_up, thumb_up]):
#         return "Fist ✊"
#     elif thumb_up and not (index_up or middle_up or ring_up or pinky_up):
#         return "Thumbs Up 👍"
#     elif not thumb_up and not index_up and middle_up and ring_up and pinky_up:
#         return "Stop ✋"
#     elif thumb_up and pinky_up and not (index_up or middle_up or ring_up):
#         return "Call Me 🤙"
#     elif thumb_up and index_up and not (middle_up or ring_up or pinky_up):
#         return "Gun Gesture 🔫"
#     elif all([index_up, middle_up]) and not (ring_up or pinky_up or thumb_up):
#         return "Give Me Food 🍽️"
#     elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
#         return "Thumbs Up 👍"
#     elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
#         return "Thumbs Down 👎"
#     else:
#         return "Unknown Gesture"

# # Emotion Detection History for Stability
# emotion_history = []

# # Video Frame Processing
# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Eye Movement Detection
#             face_results = mp_face_mesh.process(frame_rgb)
#             eye_direction = "No Face Detected"
#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     eye_direction = detect_eye_movement(face_landmarks.landmark)

#             # Emotion Detection with Smoothing
#             emotion = "Neutral"
#             try:
#                 emotion_result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='opencv', enforce_detection=False)
#                 detected_emotion = emotion_result[0]['dominant_emotion']

#                 # Add to history and maintain last 5 readings
#                 emotion_history.append(detected_emotion)
#                 if len(emotion_history) > 5:
#                     emotion_history.pop(0)

#                 # Use the most frequent emotion in last 5 frames
#                 emotion = max(set(emotion_history), key=emotion_history.count)
#             except:
#                 pass  # Ignore errors if face not detected

#             # Hand Gesture & Pointing Detection
#             hand_results = mp_hands.process(frame_rgb)
#             gesture = "No Hand Detected"
#             pointed_object = "None"
#             if hand_results.multi_hand_landmarks:
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     gesture = detect_hand_gesture(hand_landmarks.landmark)

#                     # If pointing, determine what they are pointing at
#                     if gesture == "Pointing ☝️":
#                         pointed_object = direction_to_object.get(eye_direction, "Unknown Object")

#             # Log Data
#             log_data(eye_direction, emotion, gesture, pointed_object)

#             # Display Results
#             cv2.putText(frame, f"Eye: {eye_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Gesture: {gesture}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Pointing At: {pointed_object}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():

#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/api/test', methods=['GET'])
# def test_api():
#     return {"message": "Flask API is working!"}, 200

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, render_template, Response, jsonify
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np

# app = Flask(__name__)

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# camera = cv2.VideoCapture(0)

# # Emotion Detection History
# emotion_history = []

# status_data = {
#     "Eye": "No Face Detected",
#     "Emotion": "Neutral",
#     "Gesture": "No Hand Detected",
#     "Pointing At": "None"
# }

# def detect_eye_movement(landmarks):
#     left_eye_x = (landmarks[33].x + landmarks[133].x) / 2
#     right_eye_x = (landmarks[362].x + landmarks[263].x) / 2
#     nose_x = landmarks[1].x

#     if left_eye_x > nose_x and right_eye_x > nose_x + 0.03:
#         return "Left"
#     elif left_eye_x < nose_x and right_eye_x < nose_x - 0.03:
#         return "Right"
#     else:
#         return "Center"

# def detect_hand_gesture(landmarks):
#     index_up = landmarks[8].y < landmarks[6].y
#     index_up = landmarks[8].y < landmarks[6].y
#     middle_up = landmarks[12].y < landmarks[10].y
#     ring_up = landmarks[16].y < landmarks[14].y
#     pinky_up = landmarks[20].y < landmarks[18].y
#     thumb_up = landmarks[4].y < landmarks[3].y

#     if index_up and not (middle_up or ring_up or pinky_up):
#         return "Pointing ☝️"
#     elif all([index_up, middle_up, ring_up, pinky_up]):
#         return "Open Palm 🖐️"
#     elif not any([index_up, middle_up, ring_up, pinky_up, thumb_up]):
#         return "Fist ✊"
#     elif thumb_up and not (index_up or middle_up or ring_up or pinky_up):
#         return "Thumbs Up 👍"
#     elif not thumb_up and not index_up and middle_up and ring_up and pinky_up:
#         return "Stop ✋"
#     elif thumb_up and pinky_up and not (index_up or middle_up or ring_up):
#         return "Call Me 🤙"
#     elif thumb_up and index_up and not (middle_up or ring_up or pinky_up):
#         return "Gun Gesture 🔫"
#     elif all([index_up, middle_up]) and not (ring_up or pinky_up or thumb_up):
#         return "Give Me Food 🍽️"
#     elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
#         return "Thumbs Up 👍"
#     elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
#         return "Thumbs Down 👎"
#     else:
#         return "Unknown Gesture"
    
# # Emotion Detection History for Stability
# emotion_history = []

# def generate_frames():
#     global status_data

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             face_results = mp_face_mesh.process(frame_rgb)

#             eye_direction = "No Face Detected"
#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     eye_direction = detect_eye_movement(face_landmarks.landmark)

#             emotion = "Neutral"
#             try:
#                 emotion_result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='opencv', enforce_detection=False)
#                 detected_emotion = emotion_result[0]['dominant_emotion']
#                 emotion_history.append(detected_emotion)
#                 if len(emotion_history) > 5:
#                     emotion_history.pop(0)
#                 emotion = max(set(emotion_history), key=emotion_history.count)
#             except:
#                 pass

#             hand_results = mp_hands.process(frame_rgb)
#             gesture = "No Hand Detected"
#             if hand_results.multi_hand_landmarks:
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     gesture = detect_hand_gesture(hand_landmarks.landmark)

#             status_data = {
#                 "Eye": eye_direction,
#                 "Emotion": emotion,
#                 "Gesture": gesture,
#                 "Pointing At": "None"
#             }

#             cv2.putText(frame, f"Eye: {eye_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Gesture: {gesture}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/api/get_status', methods=['GET'])
# def get_status():
#     return jsonify(status_data)

# if __name__ == "__main__":
#     app.run(debug=True)




# from flask import Flask, render_template, Response, jsonify
# import cv2
# import mediapipe as mp
# from deepface import DeepFace
# import numpy as np

# app = Flask(__name__)

# # Initialize MediaPipe
# mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# camera = cv2.VideoCapture(0)

# # Emotion Detection History
# emotion_history = []

# # Object Mapping Based on Direction
# direction_to_object = {
#     "Left": "💊 Medicine kept on the table",
#     "Right": "💧 Water bottle is there",
#     "Up": "🌀 Fan should be turned on",
#     "Down": "⬇️ Something is on the floor",
#     "Center": "Turn on the TV."
# }

# status_data = {
#     "Eye": "No Face Detected",
#     "Emotion": "Neutral",
#     "Gesture": "No Hand Detected",
#     "Pointing At": "None"
# }

# def detect_eye_movement(landmarks):
#     left_eye_x = (landmarks[33].x + landmarks[133].x) / 2
#     right_eye_x = (landmarks[362].x + landmarks[263].x) / 2
#     nose_x = landmarks[1].x

#     if left_eye_x > nose_x and right_eye_x > nose_x + 0.03:
#         return "Left"
#     elif left_eye_x < nose_x and right_eye_x < nose_x - 0.03:
#         return "Right"
#     else:
#         return "Center"

# def detect_hand_gesture(landmarks):
#     index_up = landmarks[8].y < landmarks[6].y
#     middle_up = landmarks[12].y < landmarks[10].y
#     ring_up = landmarks[16].y < landmarks[14].y
#     pinky_up = landmarks[20].y < landmarks[18].y
#     thumb_up = landmarks[4].y < landmarks[3].y

#     if index_up and not (middle_up or ring_up or pinky_up):
#         return "Pointing ☝️"
#     elif all([index_up, middle_up, ring_up, pinky_up]):
#         return "Open Palm 🖐️"
#     elif not any([index_up, middle_up, ring_up, pinky_up, thumb_up]):
#         return "Fist ✊"
#     elif thumb_up and not (index_up or middle_up or ring_up or pinky_up):
#         return "Thumbs Up 👍"
#     elif not thumb_up and not index_up and middle_up and ring_up and pinky_up:
#         return "Stop ✋"
#     elif thumb_up and pinky_up and not (index_up or middle_up or ring_up):
#         return "Call Me 🤙"
#     elif thumb_up and index_up and not (middle_up or ring_up or pinky_up):
#         return "Gun Gesture 🔫"
#     elif all([index_up, middle_up]) and not (ring_up or pinky_up or thumb_up):
#         return "Give Me Food 🍽️"
#     elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
#         return "Thumbs Up 👍"
#     elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
#         return "Thumbs Down 👎"
#     else:
#         return "Unknown Gesture"
    
# # Emotion Detection History for Stability
# emotion_history = []

# def generate_frames():
#     global status_data

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             face_results = mp_face_mesh.process(frame_rgb)

#             eye_direction = "No Face Detected"
#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     eye_direction = detect_eye_movement(face_landmarks.landmark)

#             emotion = "Neutral"
#             try:
#                 emotion_result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='opencv', enforce_detection=False)
#                 detected_emotion = emotion_result[0]['dominant_emotion']
#                 emotion_history.append(detected_emotion)
#                 if len(emotion_history) > 5:
#                     emotion_history.pop(0)
#                 emotion = max(set(emotion_history), key=emotion_history.count)
#             except:
#                 pass

#             hand_results = mp_hands.process(frame_rgb)
#             gesture = "No Hand Detected"
#             if hand_results.multi_hand_landmarks:
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     gesture = detect_hand_gesture(hand_landmarks.landmark)

#             status_data = {
#                 "Eye": eye_direction,
#                 "Emotion": emotion,
#                 "Gesture": gesture,
#                 "Pointing At": direction_to_object.get(eye_direction, "No Object Detected")
#             }

#             cv2.putText(frame, f"Eye: {eye_direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Emotion: {emotion}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#             cv2.putText(frame, f"Gesture: {gesture}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/api/get_status', methods=['GET'])
# def get_status():
#     return jsonify(status_data)

# if __name__ == "__main__":
#     app.run(debug=True)


