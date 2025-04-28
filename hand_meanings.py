import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def count_fingers(hand_landmarks, hand_type):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_type == "right":
        if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
    else:  # left hand
        if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

    # Other four fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# Start camera
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        total_fingers = 0
        gesture_text = ""

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, hand_handedness in enumerate(result.multi_handedness):
                hand_type = hand_handedness.classification[0].label.lower()
                hand_landmarks = result.multi_hand_landmarks[idx]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = count_fingers(hand_landmarks, hand_type)
                fingers_up = fingers.count(1)
                total_fingers += fingers_up

                # Detect gestures
                if fingers == [1, 0, 0, 0, 0]:
                    gesture_text = "Thumbs Up"
                elif fingers == [0, 1, 1, 0, 0]:
                    gesture_text = "Hi"
                elif fingers == [0, 0, 0, 0, 0]:
                    gesture_text = "Fist"
                elif fingers == [1, 1, 1, 1, 1]:
                    gesture_text = "Open Hand"
                else:
                    gesture_text = f"Fingers: {fingers_up}"

        # Display gesture or finger count
        cv2.putText(frame, gesture_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        cv2.imshow("Real-Time Multi Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
