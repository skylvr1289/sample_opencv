'''
An app where we can let balloons float and when Index finger croses it, it bursts
'''
import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Setup Video Capture
cap = cv2.VideoCapture(0)

# Load balloon and burst images
balloon_img = cv2.imread('balloon.png', cv2.IMREAD_UNCHANGED)
burst_img = cv2.imread('burst.png', cv2.IMREAD_UNCHANGED)

# Resize images to dqesired sizes
balloons = []
BALLOON_SPEED = 4
BALOON_FREQUENCY = 0.02
score = 0

BALLOON_RADIUS = 100
BALLOON_SIZE = (2 * BALLOON_RADIUS, 3 * BALLOON_RADIUS)  # Width, height
burst_size = (2 * BALLOON_RADIUS, 2 * BALLOON_RADIUS)  # Width, height

balloon_img = cv2.resize(balloon_img, BALLOON_SIZE, interpolation=cv2.INTER_AREA)
burst_img = cv2.resize(burst_img, burst_size, interpolation=cv2.INTER_AREA)

# Check if images have alpha channel, add if not
def ensure_alpha_channel(img):
    if img.shape[2] == 3:
        alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
        img = np.concatenate((img, alpha_channel), axis=2)
    return img

balloon_img = ensure_alpha_channel(balloon_img)
burst_img = ensure_alpha_channel(burst_img)

# Function to check if the index finger touches a balloon
def is_finger_touching_balloon(finger_tip, balloon):
    distance = np.linalg.norm(np.array(finger_tip) - np.array(balloon[:2]))
    return distance < BALLOON_RADIUS

# Function to overlay an image with transparency
def overlay_image(background, overlay, x, y):
    h, w, _ = overlay.shape

    # Check and adjust the overlay boundaries to be within the background frame
    if y < 0:
        overlay = overlay[-y:]
        h = overlay.shape[0]
        y = 0
    if y + h > background.shape[0]:
        overlay = overlay[:background.shape[0] - y]
        h = overlay.shape[0]
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if x + w > background.shape[1]:
        overlay = overlay[:, :background.shape[1] - x]
        w = overlay.shape[1]

    alpha_overlay = overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(0, 3):
        background[y:y+h, x:x+w, c] = (alpha_overlay * overlay[:, :, c] +
                                       alpha_background * background[y:y+h, x:x+w, c])

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    result = hands.process(rgb_frame)

    finger_tip = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip landmark
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            finger_tip = (int(index_finger_tip.x * w), int(index_finger_tip.y * h))

    # Add a new balloon at random intervals
    if random.random() < BALOON_FREQUENCY:  # Adjust the frequency as needed
        new_balloon = [random.randint(BALLOON_RADIUS, frame.shape[1] - BALLOON_RADIUS), frame.shape[0] + BALLOON_RADIUS, 'intact']
        balloons.append(new_balloon)

    # Move balloons upwards and check for collisions with finger
    for balloon in balloons[:]:
        if balloon[2] == 'intact':
            balloon[1] -= BALLOON_SPEED  # Adjust the speed as needed

            if finger_tip and is_finger_touching_balloon(finger_tip, balloon):
                balloon[2] = 'burst'
                balloon.append(time.time())
                score += 1
            else:
                # Draw the balloon
                overlay_image(frame, balloon_img, balloon[0] - BALLOON_RADIUS, balloon[1] - BALLOON_RADIUS)

            # Remove balloons that are out of the frame
            if balloon[1] < -BALLOON_RADIUS:
                balloons.remove(balloon)
        elif balloon[2] == 'burst':
            # Display burst image for a short duration
            overlay_image(frame, burst_img, balloon[0] - BALLOON_RADIUS, balloon[1] - BALLOON_RADIUS)
            if time.time() - balloon[3] > 0.5:  # Display burst for 0.5 seconds
                balloons.remove(balloon)

    # Display the resulting frame
    cv2.putText(frame, "Score: " + str(int(score)), (130, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.imshow('Balloon Game', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
