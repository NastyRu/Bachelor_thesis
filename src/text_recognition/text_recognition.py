import pytesseract
import numpy as np
import cv2


def get_text_from_document(path):
    image = cv2.imread(path)

    text1 = pytesseract.image_to_string(image, lang='rus')

    dilated_img = cv2.dilate(image[:, :, 1], np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(image[:, :, 1], bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    text2 = pytesseract.image_to_string(norm_img, lang='rus')
    return text1 + text2


def main():
    documents = {"passport1": 0,
                 "passport2": 1,
                 "passport3": 2,
                 "vu1": 3,
                 "vu2": 4,
                 "vu3": 5,
                 "passport": 6,
                 "visa_fra": 7,
                 "vise_deu": 8,
                 "visa_ita": 9,
                 "visa_esp": 10}

    path = ""
    text = get_text_from_document(path)


if __name__ == "__main__":
    main()
