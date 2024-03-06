import time
import cv2
from functools import lru_cache
import os
import numpy as np
from PIL import Image
from io import BytesIO


@lru_cache(maxsize=1)
def process(img) -> None:
    
    worker_pid = os.getpid()
    print(f"Handling inference request with worker PID: {worker_pid}")

    start_time = time.time()

    font = cv2.FONT_HERSHEY_SIMPLEX
    face_size = (50, 50)
    mask_font_color = (0, 255, 0)
    no_mask_font_color = (0, 0, 255)
    thickness = 2
    font_scale = 1
    mask_text = 'Mask'
    no_mask_text = 'No Mask'
    condition = None
    threshold = 80

    xml_file_path_face = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'haarcascade_frontalface_default.xml'))
    xml_file_path_mouth = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model', 'haarcascade_mcs_mouth.xml'))

    img = Image.open(BytesIO(img))
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, black_white) = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    faces = cv2.CascadeClassifier(xml_file_path_face).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces_bw = cv2.CascadeClassifier(xml_file_path_face).detectMultiScale(black_white, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0 and len(faces_bw) == 0:
        cv2.putText(img, "Face Not Found", face_size, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
        condition = "Face Not Found"

    elif len(faces) == 0 and len(faces_bw) == 1:
        cv2.putText(img, mask_text, face_size, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
        condition = mask_text

    else:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y: y+h, x: x+w]

        mouth_rects = cv2.CascadeClassifier(xml_file_path_mouth).detectMultiScale(roi_gray, 1.3, 5)

        if len(mouth_rects) == 0:
            cv2.putText(img, mask_text, face_size, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
            condition = mask_text

        else:
            for (_, my, _, _) in mouth_rects:
                if y < my < y + h:
                    cv2.putText(img, no_mask_text, face_size, font, font_scale, no_mask_font_color, thickness, cv2.LINE_AA)
                    condition = no_mask_text
                    break

    

    end_time = time.time()
    processing_time = end_time - start_time

    print(f"Inference done, worker PID: {worker_pid}")

    return condition, processing_time