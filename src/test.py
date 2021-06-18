import numpy as np
from skimage.transform import resize
import matplotlib.image as mpimg
from googlenet.inception_v1 import InceptionV1
from text_recognition.text_recognition import get_all_text
from text_recognition.train import text_preprocessing
import pickle
import glob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


w1 = [1, 1.185, 1, 1.1, 1.025, 0.1, 0.188, 1, 0.3, 0.138, 0.4]
w2 = [1.1, 1.075, 1, 1.026, 1.1, 1.1, 1.125, 1.15, 1.5, 3, 1.5]


def prediction(path, model, tfidf_vect, SVM, labelencode):
    img = mpimg.imread(path)
    print(path)
    img = resize(img, (224, 224, 3))
    img = img.reshape(1, 224, 224, 3)
    out = model.predict(img)

    predicted_label_visual = np.argmax(out[2])
    predicted_proba_visual = out[2][0][predicted_label_visual] * \
                             w1[predicted_label_visual]

    f = open("research_data_text" + path[13:-4] + "txt", 'r',
             encoding='utf-8', errors='ignore')
    text = str(f.readlines())
    f.close()
    text_processed = text_preprocessing(text)
    text_processed_vectorized = tfidf_vect.transform([text_processed])

    prediction_SVM = SVM.predict(text_processed_vectorized)
    predicted_label_text = labelencode.inverse_transform(prediction_SVM)[0]
    predicted_proba_text = SVM.predict_proba(text_processed_vectorized)[0][predicted_label_text] * \
                           w2[predicted_label_text]

    if (predicted_proba_text > predicted_proba_visual):
        npredicted_label_text = predicted_label_text
        return predicted_label_visual, npredicted_label_text, predicted_label_text
    else:
        return predicted_label_visual, predicted_label_text, predicted_label_visual


def get_results(label, path, final_visual, final_text, final,
                test, model, tfidf_vect, SVM, labelencode):
    images = glob.glob(path)
    for image in images:
        get_label = prediction(image, model, tfidf_vect, SVM, labelencode)
        final_visual.append(get_label[0])
        final_text.append(get_label[1])
        final.append(get_label[2])
        test.append(label)
    return final_visual, final_text, final, test


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
    final_visual = []
    final_text = []
    test = []
    print('passport')
    final_visual, final_text, final, test = get_results(0, 'research_data/passport/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('passport1')
    final_visual, final_text, final, test = get_results(1, 'research_data/passport1/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('passport2')
    final_visual, final_text, final, test = get_results(2, 'research_data/passport2/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('passport3')
    final_visual, final_text, final, test = get_results(3, 'research_data/passport3/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('vu1')
    final_visual, final_text, final, test = get_results(4, 'research_data/vu1/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('vu2')
    final_visual, final_text, final, test = get_results(5, 'research_data/vu2/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('vu3')
    final_visual, final_text, final, test = get_results(6, 'research_data/vu3/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_deu')
    final_visual, final_text, final, test = get_results(7, 'research_data/visa_deu/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_esp')
    final_visual, final_text, final, test = get_results(8, 'research_data/visa_esp/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_fra')
    final_visual, final_text, final, test = get_results(9, 'research_data/visa_fra/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)
    print('visa_ita')
    final_visual, final_text, final, test = get_results(10, 'research_data/visa_ita/*.*',
                              final_visual, final_text, final, test, model, tfidf_vect, SVM, labelencode)

    labels = ['Загранпаспорт', 'Паспорт, страница1', 'Паспорт, страница2',
              'Паспорт, прописка', 'ВУ, образец1', 'ВУ, образец2',
              'ВУ, образец3', 'Виза Германия', 'Виза Испания',
              'Виза Франция', 'Виза Италия']

    mat = confusion_matrix(test, final)
    ax = plt.subplot()
    sns.heatmap(mat, annot=True, fmt='g', ax=ax, cmap = plt.get_cmap('Blues'))
    ax.set_xlabel('Предсказанные')
    ax.set_ylabel('Реальные')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

    mat = confusion_matrix(test, final_text)
    ax = plt.subplot()
    sns.heatmap(mat, annot=True, fmt='g', ax=ax, cmap = plt.get_cmap('Blues'))
    ax.set_xlabel('Предсказанные')
    ax.set_ylabel('Реальные')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

    mat = confusion_matrix(test, final_visual)
    ax = plt.subplot()
    sns.heatmap(mat, annot=True, fmt='g', ax=ax, cmap = plt.get_cmap('Blues'))
    ax.set_xlabel('Предсказанные')
    ax.set_ylabel('Реальные')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

    y1 = get_f1_measure(11, final_visual, test)
    y2 = get_f1_measure(11, final_text, test)
    y3 = get_f1_measure(11, final, test)
    plt.xlabel("Классы")
    plt.ylabel("F1-мера")
    plt.xticks(rotation=45)
    plt.plot(labels, y1, label="Визуальный")
    plt.plot(labels, y2, label="Текстовый")
    plt.plot(labels, y3, label="Ансамбль")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
