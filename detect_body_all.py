import cv2
import mediapipe as mp

# Initialize MediaPipe models
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize Hands, Pose and Face Detection models
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Initialize OpenCV to capture video
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands, Pose and Face Detection
    hand_results = hands.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    # Draw Hand Landmarks
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw Body Pose Landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Draw Face Detection
    if face_results.detections:
        for detection in face_results.detections:
            # Draw bounding box and landmarks for face detection
            mp_drawing.draw_detection(frame, detection)

    # Display the frame with hand, body pose, and face detection
    cv2.imshow('Hand, Body, and Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
