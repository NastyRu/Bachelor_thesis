import cv2


def bw_image(path):
    im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


def get_visa_class(path):
    image = bw_image(path)
    visa = bw_image("documents_keypoints/visa.png")  # fra
    visum = bw_image("documents_keypoints/visum.png")  # deu
    visto = bw_image("documents_keypoints/visto.png")  # ita
    visado = bw_image("documents_keypoints/visado.png")  # esp

    visto_res = cv2.matchTemplate(image, visto, cv2.TM_CCOEFF_NORMED)
    visum_res = cv2.matchTemplate(image, visum, cv2.TM_CCOEFF_NORMED)
    visado_res = cv2.matchTemplate(image, visado, cv2.TM_CCOEFF_NORMED)
    visa_res = cv2.matchTemplate(image, visa, cv2.TM_CCOEFF_NORMED)

    visas = {7: visto_res, 8: visum_res, 9: visado_res, 10: visa_res}

    max_key = max(visas, key=visas.get)
    return max_key


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
    visa_class = get_visa_class(path)


if __name__ == "__main__":
    main()
