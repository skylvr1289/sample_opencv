import cv2
import mediapipe as mp
import math
import numpy as np
# use pycaw  library to control audio in windows, osascript is fro mac
import osascript

cv2.namedWindow("cloud_geek")
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


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
                cv2.circle(frame, (thumb_cx, thumb_cy), 20, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (index_cx, index_cy), 20, (255, 0, 255), cv2.FILLED)

                # draw a line
                cv2.line(frame, (thumb_cx, thumb_cy), (index_cx, index_cy), (255, 0, 255), 5)
                distance_thumb_index = math.hypot(index_cx - thumb_cx, index_cy - thumb_cy)
                distance_index_middletip = math.hypot(index_cx - middletip_cx, index_cy - middletip_cy)

                #print(distance_thumb_index)
                # set audio value
                vol_value = int(np.interp(distance_thumb_index, [50, 560], [0, 100]))

                vol = "set volume output volume " + str(vol_value)  # don't change this str text
                osascript.osascript(str(vol))
                # draw the volume bar (1 rectangle, and 1 filled rectangle with varying x length)
                cv2.rectangle(frame, (50, 60), (450, 80), (0, 0, 255), 20)
                # value of vol bar will be between x coordinates or rectangle [60-450]
                vol_bar = int(np.interp(distance_thumb_index, [50, 560], [60, 450]))
                cv2.rectangle(frame, (50, 60), (vol_bar, 80), (225, 255, 255), cv2.FILLED)
                cv2.putText(frame, str(vol_value), (500, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
                print(distance_index_middletip)
                # for closing. bring index and middle finger tips closer.
                if distance_index_middletip < 120:
                    cv2.circle(frame, (middletip_cx, middletip_cy), 20, (0, 0, 255), cv2.FILLED)
                    cv2.line(frame, (middletip_cx, middletip_cy), (index_cx, index_cy), (0, 0, 255), 5)
                    cv2.putText(frame, str(int(distance_index_middletip)), (middletip_cx+30, middletip_cy-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
                    cv2.imshow("cloud_geek", frame)
                    if distance_index_middletip <50:
                        exit(0)

        cv2.imshow("cloud_geek", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
