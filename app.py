import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

gesture_names = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}

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

    return fingers.count(1)

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

        if result.multi_hand_landmarks and result.multi_handedness:
            for idx, hand_handedness in enumerate(result.multi_handedness):
                hand_type = hand_handedness.classification[0].label.lower()
                hand_landmarks = result.multi_hand_landmarks[idx]

                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers_up = count_fingers(hand_landmarks, hand_type)
                total_fingers += fingers_up

        # Display total fingers count
        cv2.putText(frame, f"Count: {total_fingers}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Real-Time Multi Hand Gesture Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
