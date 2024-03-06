import cv2
import numpy as np

def detect_mask(namefile, file) -> str:
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_size = (30, 30)
    mask_font_color = (0, 255, 0)
    no_mask_font_color = (0, 0, 255)
    thickness = 2
    font_scale = 1
    mask_text = 'Mask'
    no_mask_text = 'No Mask'
    condition = None
    threshold = 80

    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, black_white) = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    faces = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)
    faces_bw = cv2.CascadeClassifier('model/haarcascade_frontalface_default.xml').detectMultiScale(black_white, 1.1, 4)

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

        mouth_rects = cv2.CascadeClassifier('model/haarcascade_mcs_mouth.xml').detectMultiScale(roi_gray, 1.3, 5)

        if len(mouth_rects) == 0:
            cv2.putText(img, mask_text, face_size, font, font_scale, mask_font_color, thickness, cv2.LINE_AA)
            condition = mask_text

        else:
            for (_, my, _, _) in mouth_rects:
                if y < my < y + h:
                    cv2.putText(img, no_mask_text, face_size, font, font_scale, no_mask_font_color, thickness, cv2.LINE_AA)
                    condition = no_mask_text
                    break

    cv2.imwrite(namefile, img)
    return condition
