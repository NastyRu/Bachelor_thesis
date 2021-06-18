from scipy.optimize import minimize
import numpy as np
from skimage.transform import resize
import matplotlib.image as mpimg
from googlenet.inception_v1 import InceptionV1
from text_recognition.train import text_preprocessing
import pickle
import glob
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def prediction(path, model, tfidf_vect, SVM, labelencode, w1, w2):
    img = mpimg.imread(path)
    print(path)
    img = resize(img, (224, 224, 3))
    img = img.reshape(1, 224, 224, 3)
    out = model.predict(img)

    predicted_label_visual = np.argmax(out[2])
    predicted_proba_visual = out[2][0][predicted_label_visual] * \
                             w1[predicted_label_visual]

    f = open("text_recognition" + path[9:-4] + "txt", 'r', encoding='utf-8', errors='ignore')
    text = str(f.readlines())
    f.close()
    text_processed = text_preprocessing(text)
    text_processed_vectorized = tfidf_vect.transform([text_processed])

    prediction_SVM = SVM.predict(text_processed_vectorized)
    predicted_label_text = labelencode.inverse_transform(prediction_SVM)[0]
    predicted_proba_text = SVM.predict_proba(text_processed_vectorized)[0][predicted_label_text] * \
                           w2[predicted_label_text]

    if (predicted_proba_text > predicted_proba_visual):
        return predicted_label_text
    else:
        return predicted_label_visual


def get_results(label, path, final, test, model,
                tfidf_vect, SVM, labelencode, w1, w2):
    i = 0
    images = glob.glob(path)
    for image in images:
        get_label = prediction(image, model, tfidf_vect,
                               SVM, labelencode, w1, w2)
        final.append(get_label)
        test.append(label)
        i += 1
        if (i > len(images) * 0.2):
            break
    return final, test


def get_f1_measure(class_num, y_final, y_test):
    f1_tp = [0] * class_num
    f1_fp = [0] * class_num
    f1_fn = [0] * class_num
    for i in range(len(y_final)):
        if y_final[i] == y_test[i]:
            f1_tp[y_final[i]] += 1
        else:
            f1_fn[y_test[i]] += 1
            f1_fp[y_final[i]] += 1

    f1 = [0] * class_num
    for i in range(class_num):
        try:
            p = f1_tp[i] / (f1_tp[i] + f1_fp[i])
        except:
            p = 0
        try:
            r = f1_tp[i] / (f1_tp[i] + f1_fn[i])
        except:
            r = 0
        try:
            f1[i] = 2 * p * r / (p + r)
        except:
            f1[i] = 0
    return f1


def result(w1, w2):
    model_name = 'googlenet_weights.h5'
    model = InceptionV1().architecture()
    model.load_weights(model_name)

    labelencode = pickle.load(open('labelencoder_fitted.pkl', 'rb'))
    tfidf_vect = pickle.load(open('Tfidf_vect_fitted.pkl', 'rb'))
    SVM = pickle.load(open('svm_trained_model.sav', 'rb'))

    final = []
    test = []
    print('passport')
    final, test = get_results(0, 'googlenet/training_data/passport/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('passport1')
    final, test = get_results(1, 'googlenet/training_data/passport1/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('passport2')
    final, test = get_results(2, 'googlenet/training_data/passport2/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('passport3')
    final, test = get_results(3, 'googlenet/training_data/passport3/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('vu1')
    final, test = get_results(4, 'googlenet/training_data/vu1/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('vu2')
    final, test = get_results(5, 'googlenet/training_data/vu2/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('vu3')
    final, test = get_results(6, 'googlenet/training_data/vu3/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('visa_deu')
    final, test = get_results(7, 'googlenet/training_data/visa_deu/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('visa_esp')
    final, test = get_results(8, 'googlenet/training_data/visa_esp/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('visa_fra')
    final, test = get_results(9, 'googlenet/training_data/visa_fra/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)
    print('visa_ita')
    final, test = get_results(10, 'googlenet/training_data/visa_ita/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode,
                              w1, w2)

    return get_f1_measure(11, final, test)


def get_result(args):
    w1 = args[:11]
    w2 = args[11:]
    f1 = result(w1, w2)
    return sum(f1) / len(f1)


if __name__ == "__main__":
    initial_guess = [0] * 22
    result = minimize(get_result, initial_guess, method='powell')
    print(result.x)
