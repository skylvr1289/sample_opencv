import cv2
import mediapipe as mp
import math

import pyautogui

cv2.namedWindow("cloud_geek")
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
space_pressed=False
'''
# steps to run
1. Make sure to open chrome. Turn off your wifi so that you can run T-rex game in chrome. 
2. run this python code. 
4. with your mouse cursor click on the chrome window
5. now start increasing the distance between thumb and index finger tips to make a jump.
'''
while True:
    success, frame = cap.read()
    frame = cv2.flip(frame,1)
    h, w, c = frame.shape
    if frame is not None:
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                lmPoints = handLms.landmark
                thumb = lmPoints[4]
                index = lmPoints[8]
                middletip = lmPoints[12]

                thumb_cx, thumb_cy = int(thumb.x * w), int(thumb.y * h)
                index_cx, index_cy = int(index.x * w), int(index.y * h)
                middletip_cx, middletip_cy =  int(middletip.x * w), int(middletip.y * h)

                ## draw circles
                # cv2.circle(frame, (thumb_cx, thumb_cy), 20, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (index_cx, index_cy), 20, (0, 255, 0), cv2.FILLED)

                # draw a line
                # cv2.line(frame, (thumb_cx, thumb_cy), (index_cx, index_cy), (255, 0, 255), 5)
                distance_thumb_index = math.hypot(index_cx - thumb_cx, index_cy - thumb_cy)
                print(distance_thumb_index)
                # for closing. bring index and middle finger tips closer.
                ## press space bar when distance is more
                if distance_thumb_index > 200:
                    if not(space_pressed):
                        print("press")
                        pyautogui.keyDown("space")
                        space_pressed=True
                else:
                    if space_pressed:
                        print("lift")
                        space_pressed=False
                        pyautogui.keyUp("space")

        cv2.imshow("cloud_geek", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
