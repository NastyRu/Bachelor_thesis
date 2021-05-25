import numpy as np
from skimage.transform import resize
import matplotlib.image as mpimg
from googlenet.inception_v1 import InceptionV1
from text_recognition.text_recognition import get_all_text
from text_recognition.train import text_preprocessing
import pickle


def main():
    path = "10.jpeg"
    
    model_name = 'googlenet_weights.h5'
    model = InceptionV1().architecture()
    model.load_weights(model_name)

    img = mpimg.imread(path)
    img = resize(img, (224, 224, 3))
    img = img.reshape(1, 224, 224, 3)
    out = model.predict(img)

    predicted_label = np.argmax(out[2])
    print('Predicted Class: ', predicted_label)

    labelencode = pickle.load(open('labelencoder_fitted.pkl', 'rb'))
    tfidf_vect = pickle.load(open('Tfidf_vect_fitted.pkl', 'rb'))
    SVM = pickle.load(open('svm_trained_model.sav', 'rb'))

    text = get_all_text(path)
    text_processed = text_preprocessing(text)
    text_processed_vectorized = tfidf_vect.transform([text_processed])

    prediction_SVM = SVM.predict(text_processed_vectorized)

    print("Prediction from SVM Model:", labelencode.inverse_transform(prediction_SVM)[0])


if __name__ == "__main__":
    main()
