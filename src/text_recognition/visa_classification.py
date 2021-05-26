import cv2


def bw_image(path):
    im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


def get_visa_class(path):
    image = bw_image(path)
    visum = bw_image("documents_keypoints/visum.png")  # deu
    visado = bw_image("documents_keypoints/visado.png")  # esp
    visa = bw_image("documents_keypoints/visa.png")  # fra
    visto = bw_image("documents_keypoints/visto.png")  # ita

    _, visto_res, _, _ = cv2.minMaxLoc(cv2.matchTemplate(image, visto,
                                                         cv2.TM_CCOEFF_NORMED))
    _, visum_res, _, _ = cv2.minMaxLoc(cv2.matchTemplate(image, visum,
                                                         cv2.TM_CCOEFF_NORMED))
    _, visado_res, _, _ = cv2.minMaxLoc(cv2.matchTemplate(image, visado,
                                                          cv2.TM_CCOEFF_NORMED))
    _, visa_res, _, _ = cv2.minMaxLoc(cv2.matchTemplate(image, visa,
                                                        cv2.TM_CCOEFF_NORMED))

    visas = {7: visum_res, 8: visado_res, 9: visa_res, 10: visto_res}

    return max(visas, key=visas.get), max(visas.values())
