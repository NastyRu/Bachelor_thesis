import numpy as np
from skimage.transform import resize
import matplotlib.image as mpimg
from googlenet.inception_v1 import InceptionV1


def main():
    model_name = 'googlenet_weights.h5'

    model = InceptionV1().architecture()
    model.summary()
    model.load_weights(model_name)

    img = mpimg.imread('example.jpg')
    img = resize(img, (224, 224, 3))
    img = img.reshape(1, 224, 224, 3)
    out = model.predict(img)

    predicted_label = np.argmax(out[2])
    print('Predicted Class: ', predicted_label)


if __name__ == "__main__":
    main()
