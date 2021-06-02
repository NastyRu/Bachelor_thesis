import numpy as np
from skimage.transform import resize
import matplotlib.image as mpimg
from googlenet.inception_v1 import InceptionV1
from text_recognition.text_recognition import get_all_text
from text_recognition.train import text_preprocessing
import pickle


class PredictionClass():
    def __init__(self):
        self.model = InceptionV1().architecture()
        self.model.load_weights('googlenet_weights.h5')

        self.labelencode = pickle.load(open('labelencoder_fitted.pkl', 'rb'))
        self.tfidf_vect = pickle.load(open('Tfidf_vect_fitted.pkl', 'rb'))
        self.SVM = pickle.load(open('svm_trained_model.sav', 'rb'))

        self.documents = {0: "Загранпаспорт",
                          1: "Страница паспорта с информацией о выдаче",
                          2: "Страница паспорта с личными данными",
                          3: "Страница паспорта с пропиской",
                          4: "Водительское удостоверение, бумажное, 1995-2011",
                          5: "Водительское удостоверение, пластиковое, 1995-2011",
                          6: "Водительское удостоверение, новое",
                          7: "Шенгенская виза, Германия",
                          8: "Шенгенская виза, Испания",
                          9: "Шенгенская виза, Франция",
                          10: "Шенгенская виза, Италия"}

        self.w1 = [0.09, 0.185, 1, 1.1, 1.025, 0.1, 0.188, 1.1, 1.075, 0.138, 1.025]
        self.w2 = [1.1, 1.075, 1, 1.026, 1.1, 1.1, 1.125, 1.05, 0.8, 1.1, 1.125]

    def predict_class(self, path):
        img = mpimg.imread(path)
        img = resize(img, (224, 224, 3))
        img = img.reshape(1, 224, 224, 3)
        out = self.model.predict(img)

        predicted_label_visual = np.argmax(out[2])
        predicted_proba_visual = out[2][0][predicted_label_visual] * \
                                 self.w1[predicted_label_visual]

        text = get_all_text(path)
        text_processed = text_preprocessing(text)
        text_processed_vectorized = self.tfidf_vect \
                                        .transform([text_processed])

        prediction_SVM = self.SVM.predict(text_processed_vectorized)
        predicted_label_text = self.labelencode \
                                   .inverse_transform(prediction_SVM)[0]
        predicted_proba_text = self.SVM.predict_proba(
                               text_processed_vectorized)[0] \
                               [predicted_label_text] * \
                               self.w2[predicted_label_text]

        if (predicted_proba_text > predicted_proba_visual):
            predicted_label = predicted_label_text
        else:
            predicted_label = predicted_label_visual

        return self.documents.get(predicted_label)
