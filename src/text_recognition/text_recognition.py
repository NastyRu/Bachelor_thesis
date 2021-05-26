import pytesseract
import numpy as np
import cv2


def get_text_from_document(path, lang):
    image = cv2.imread(path)

    text1 = pytesseract.image_to_string(image, lang=lang)

    dilated_img = cv2.dilate(image[:, :, 1], np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(image[:, :, 1], bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    text2 = pytesseract.image_to_string(norm_img, lang=lang)
    return text1 + text2


def get_all_text(path):
    text1 = get_text_from_document(path, 'rus')
    text2 = get_text_from_document(path, 'eng')
    return text1 + text2
