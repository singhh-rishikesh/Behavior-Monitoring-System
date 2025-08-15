
from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe
mp_hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

camera = cv2.VideoCapture(0)

# Emotion Detection History
emotion_history = []

# Object Mapping Based on Direction and Gesture
direction_to_object = {
    "Left": "💊 Medicine kept on the table",
    "Right": "💧 Water bottle is there",
    "Up": "🌀 Fan should be turned on",
    "Down": "⬇️ Something is on the floor",
    "Center": "Turn on the TV."
}

gesture_to_object = {
    "Pointing Up ☝️": "🌀 Fan should be turned on",
    "Pointing Down 👇": "⬇️ Something is on the floor",
    "Pointing Left 👈": "💊 Medicine kept on the table",
    "Pointing Right 👉": "💧 Water bottle is there",
    "Open Palm 🖐️": "🤚 Stop or wave detected",
    "Fist ✊": "👊 Ready for action",
    "Thumbs Up 👍": "👍 Positive acknowledgment",
    "Thumbs Down 👎": "👎 ⬇️ Something is on the floor",
    "Call Me 🤙": "📞 Call gesture detected",
    "Gun Gesture 🔫": "🔫 Shooter gesture detected",
    "Give Me Food 🍽️": "🍽️ Food request detected",
    "Stop ✋": "✋ Stop signal detected"
}

status_data = {
    "Eye": "No Face Detected",
    "Emotion": "Neutral",
    "Gesture": "No Hand Detected",
    "Eye_responce": "None",
    "Pointing_At": "None"
}

def detect_eye_movement(landmarks):
    left_eye_x = (landmarks[33].x + landmarks[133].x) / 2
    right_eye_x = (landmarks[362].x + landmarks[263].x) / 2
    nose_x = landmarks[1].x

    if left_eye_x > nose_x and right_eye_x > nose_x + 0.03:
        return "Left"
    elif left_eye_x < nose_x and right_eye_x < nose_x - 0.03:
        return "Right"
    else:
        return "Center"

def detect_pointing_gesture(landmarks):
    # Index finger position
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    
    # Other fingers
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Check if other fingers are closed (their y position should be greater than their base)
    other_fingers_closed = (
        middle_tip.y > landmarks[10].y and  # Middle finger closed
        ring_tip.y > landmarks[14].y and    # Ring finger closed
        pinky_tip.y > landmarks[18].y       # Pinky closed
    )
    
    # Determine pointing direction
    if other_fingers_closed:
        # Pointing up
        if index_tip.y < index_pip.y - 0.1:
            return "Pointing Up ☝️"
        # Pointing down
        elif index_tip.y > index_pip.y + 0.1:
            return "Pointing Down 👇"
        # Pointing left
        elif index_tip.x < index_pip.x - 0.1:
            return "Pointing Left 👈"
        # Pointing right
        elif index_tip.x > index_pip.x + 0.1:
            return "Pointing Right 👉"
    
    return detect_other_gestures(landmarks)

def detect_other_gestures(landmarks):
    # All fingertips
    tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
    pips = [landmarks[3], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
    
    # Check if all fingers are extended
    all_extended = all(tip.y < pip.y for tip, pip in zip(tips[1:], pips[1:]))
    
    if all_extended:
        return "Open Palm 🖐️"
    
    # Check for fist (all fingers closed)
    all_closed = all(tip.y > pip.y for tip, pip in zip(tips[1:], pips[1:]))
    if all_closed:
        return "Fist ✊"
    
    # Thumbs up/down
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_extended = landmarks[8].y < landmarks[6].y
    
    if not index_extended and thumb_tip.y < thumb_ip.y:
        return "Thumbs Up 👍"
    elif not index_extended and thumb_tip.y > thumb_ip.y:
        return "Thumbs Down 👎"
    
    return "Unknown Gesture"




def detect_hand_gesture(landmarks):
    # Index Finger Position Relative to Joint
    index_up = landmarks[8].y < landmarks[6].y
    index_down = landmarks[8].y > landmarks[6].y
    index_left = landmarks[8].x < landmarks[6].x
    index_right = landmarks[8].x > landmarks[6].x

    # Other Fingers Status
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    thumb_up = landmarks[4].y < landmarks[3].y

    # **Wrist Orientation Check**
    wrist_x = landmarks[0].x  # Wrist X Position
    wrist_y = landmarks[0].y  # Wrist Y Position
    thumb_x = landmarks[4].x  # Thumb X Position

    palm_facing_up = thumb_x > wrist_x  # Thumb right -> Palm up
    palm_facing_down = thumb_x < wrist_x  # Thumb left -> Palm down

    # **Pointing Gestures**
    if index_up and palm_facing_up and not (middle_up or ring_up or pinky_up):
        return "Pointing Up ☝️ (Palm Side Up)"
    
    elif index_down and palm_facing_down and not (middle_up or ring_up or pinky_up):
        return "Pointing Down 👇 (Palm Facing Opposite)"
    
    elif index_left and palm_facing_down and not (middle_up or ring_up or pinky_up):
        return "Pointing Left 👈 (Palm Facing Opposite)"
    
    elif index_right and palm_facing_down and not (middle_up or ring_up or pinky_up):
        return "Pointing Right 👉 (Palm Facing Opposite)"
    
    elif all([index_up, middle_up, ring_up, pinky_up]):
        return "Open Palm 🖐️"
    
    elif not any([index_up, middle_up, ring_up, pinky_up, thumb_up]):
        return "Fist ✊"
    
    elif thumb_up and not (index_up or middle_up or ring_up or pinky_up):
        return "Thumbs Up 👍"
    
    elif not thumb_up and index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Down 👎"
    
    elif not thumb_up and not index_up and middle_up and ring_up and pinky_up:
        return "Stop ✋"
    
    elif thumb_up and pinky_up and not (index_up or middle_up or ring_up):
        return "Call Me 🤙"
    
    elif thumb_up and index_up and not (middle_up or ring_up or pinky_up):
        return "Gun Gesture 🔫"
    
    elif all([index_up, middle_up]) and not (ring_up or pinky_up or thumb_up):
        return "Give Me Food 🍽️"
    
    else:
        return "Unknown Gesture"

def draw_text_with_background(frame, text, position, font_scale=0.8, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle coordinates
    padding = 5
    bg_coords = (
        position[0],
        position[1] - text_height - padding,
        position[0] + text_width + padding * 2,
        position[1] + padding
    )
    
    # Draw background rectangle
    cv2.rectangle(frame, (bg_coords[0], bg_coords[1]), (bg_coords[2], bg_coords[3]), (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, (position[0] + padding, position[1]), font, font_scale, (255, 255, 255), thickness)

def generate_frames():
    global status_data

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = mp_face_mesh.process(frame_rgb)

            eye_direction = "No Face Detected"
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    eye_direction = detect_eye_movement(face_landmarks.landmark)

            emotion = "Neutral"
            try:
                emotion_result = DeepFace.analyze(frame, actions=['emotion'], detector_backend='opencv', enforce_detection=False)
                detected_emotion = emotion_result[0]['dominant_emotion']
                emotion_history.append(detected_emotion)
                if len(emotion_history) > 5:
                    emotion_history.pop(0)
                emotion = max(set(emotion_history), key=emotion_history.count)
            except:
                pass

            hand_results = mp_hands.process(frame_rgb)
            gesture = "No Hand Detected"
            pointing_gesture = "No Pointing Detected"

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                    pointing_gesture = detect_pointing_gesture(hand_landmarks.landmark)
                    gesture = pointing_gesture

            # Update status data with correct gesture mapping
            status_data = {
                "Eye": eye_direction,
                "Emotion": emotion,
                "Gesture": gesture,
                "Eye_responce": direction_to_object.get(eye_direction, "No Direction Detected"),
                "Pointing_At": gesture_to_object.get(gesture, "No Object Detected")
            }

            # Draw status on frame with background
            y_position = 40
            draw_text_with_background(frame, f"Eye: {eye_direction}", (20, y_position))
            y_position += 40
            draw_text_with_background(frame, f"Emotion: {emotion}", (20, y_position))
            y_position += 40
            draw_text_with_background(frame, f"Gesture: {gesture}", (20, y_position))
            y_position += 40
            draw_text_with_background(frame, f"Object: {status_data['Pointing_At']}", (20, y_position))

            # Draw direction arrows based on eye direction
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            arrow_length = 100
            arrow_color = (0, 255, 0)
            arrow_thickness = 2

            if eye_direction == "Left":
                cv2.arrowedLine(frame, center, (center[0] - arrow_length, center[1]), arrow_color, arrow_thickness)
            elif eye_direction == "Right":
                cv2.arrowedLine(frame, center, (center[0] + arrow_length, center[1]), arrow_color, arrow_thickness)
            elif eye_direction == "Up":
                cv2.arrowedLine(frame, center, (center[0], center[1] - arrow_length), arrow_color, arrow_thickness)
            elif eye_direction == "Down":
                cv2.arrowedLine(frame, center, (center[0], center[1] + arrow_length), arrow_color, arrow_thickness)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/get_status', methods=['GET'])
def get_status():
    return jsonify(status_data)

if __name__ == "__main__":
    app.run(debug=True)