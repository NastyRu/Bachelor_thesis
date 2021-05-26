import numpy as np
from skimage.transform import resize
import matplotlib.image as mpimg
from googlenet.inception_v1 import InceptionV1
from text_recognition.text_recognition import get_all_text
from text_recognition.train import text_preprocessing
import pickle
import glob
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def prediction(path, model, tfidf_vect, SVM, labelencode):
    img = mpimg.imread(path)
    img = resize(img, (224, 224, 3))
    img = img.reshape(1, 224, 224, 3)
    out = model.predict(img)

    predicted_label = np.argmax(out[2])

    if (predicted_label == 7):
        text = get_all_text(path)
        text_processed = text_preprocessing(text)
        text_processed_vectorized = tfidf_vect.transform([text_processed])

        prediction_SVM = SVM.predict(text_processed_vectorized)
        predicted_label = labelencode.inverse_transform(prediction_SVM)[0]

    return predicted_label


def get_results(label, path, final, test, model, tfidf_vect, SVM, labelencode):
    i = 0
    images = glob.glob(path)
    for image in images:
        get_label = prediction(image, model, tfidf_vect, SVM, labelencode)
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


def main():
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
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('passport1')
    final, test = get_results(1, 'googlenet/training_data/passport1/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('passport2')
    final, test = get_results(2, 'googlenet/training_data/passport2/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('passport3')
    final, test = get_results(3, 'googlenet/training_data/passport3/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('vu1')
    final, test = get_results(4, 'googlenet/training_data/vu1/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('vu2')
    final, test = get_results(5, 'googlenet/training_data/vu2/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('vu3')
    final, test = get_results(6, 'googlenet/training_data/vu3/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_deu')
    final, test = get_results(7, 'googlenet/training_data/visa_deu/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_esp')
    final, test = get_results(8, 'googlenet/training_data/visa_esp/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_fra')
    final, test = get_results(9, 'googlenet/training_data/visa_fra/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_ita')
    final, test = get_results(10, 'googlenet/training_data/visa_ita/*.*',
                              final, test, model, tfidf_vect, SVM, labelencode)

    mat = confusion_matrix(test, final)
    plot_confusion_matrix(mat, figsize=(9, 9), colorbar=True)
    plt.show()

    print(get_f1_measure(11, final, test))


if __name__ == "__main__":
    main()
