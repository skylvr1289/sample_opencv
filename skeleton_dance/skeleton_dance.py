import cv2
import mediapipe as mp
import numpy as np

mp_draw = mp.solutions.drawing_utils
mp_draw_style = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

cv2.namedWindow("cloud_geek")
# cap = cv2.VideoCapture(0) # to capture from camera
cap = cv2.VideoCapture('srivalli3.mov')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')


out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width,frame_height),True)
while True:
    with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            refine_face_landmarks=True) as holistic:
        success, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        #h, w, c = frame.shape

        black_img =   np.zeros((frame_height, frame_width, 3), dtype = np.uint8)
        if frame is not None:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(imgRGB)
            mp_draw.draw_landmarks(black_img, results.face_landmarks )
            mp_draw.draw_landmarks(black_img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        cv2.imshow("preview", black_img)
        cv2.imshow("2", frame)

        out.write(black_img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
print('releasing')
cv2.destroyAllWindows()