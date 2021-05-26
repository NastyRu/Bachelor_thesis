import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import glob
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


STOPWORDS = stopwords.words('english') + \
            stopwords.words('russian') + \
            stopwords.words('german')  + \
            stopwords.words('french')  + \
            stopwords.words('italian') + \
            stopwords.words('spanish')


def text_preprocessing(text):
    text = text.lower()
    text_words_list = word_tokenize(text)

    clear_words = []
    for word in text_words_list:
        if word not in STOPWORDS:
            clear_words.append(word)

    return str(clear_words)


def make_dataframe(data, labels):
    corpus = pd.DataFrame({'text': np.array(data), 'label': np.array(labels)})

    corpus['text'] = corpus['text'].map(lambda x: str(x))
    corpus['text'].dropna(inplace=True)
    corpus['text'] = [entry.lower() for entry in corpus['text']]
    corpus['text'] = [word_tokenize(entry) for entry in corpus['text']]
    for index, words in enumerate(corpus['text']):
        clear_words = []
        for word in words:
            if word not in STOPWORDS:
                clear_words.append(word)
        corpus.loc[index, 'text_final'] = str(clear_words)

    return corpus


def make_train_data(class_type, class_dir, data, labels):
    for i in class_dir:
        f = open(i, 'r', encoding='utf-8', errors='ignore')
        data.append(f.readlines())
        f.close()
        labels.append(class_type)
    return data, labels


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
    data = []
    labels = []

    data, labels = make_train_data(7, glob.glob('training_data/visa_deu/*.*'),
                                   data, labels)
    data, labels = make_train_data(8, glob.glob('training_data/visa_esp/*.*'),
                                   data, labels)
    data, labels = make_train_data(9, glob.glob('training_data/visa_fra/*.*'),
                                   data, labels)
    data, labels = make_train_data(10, glob.glob('training_data/visa_ita/*.*'),
                                   data, labels)

    corpus = make_dataframe(data, labels)

    x_train, x_test, y_train, y_test = train_test_split(corpus['text_final'],
                                                        corpus['label'],
                                                        test_size=0.3,
                                                        random_state=36)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    tfidf_vect = TfidfVectorizer(max_features=5000)
    tfidf_vect.fit(corpus['text_final'])

    x_train_tfidf = tfidf_vect.transform(x_train)
    x_test_tfidf = tfidf_vect.transform(x_test)

    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(x_train_tfidf, y_train)

    pickle.dump(encoder, open('labelencoder_fitted.pkl', 'wb'))
    pickle.dump(tfidf_vect, open('tfidf_vect_fitted.pkl', 'wb'))
    pickle.dump(SVM, open('svm_trained_model.sav', 'wb'))

    predictions_SVM = SVM.predict(x_test_tfidf)

    print("Accuracy Score: ", accuracy_score(predictions_SVM, y_test) * 100)

    mat = confusion_matrix(y_test, predictions_SVM)
    plot_confusion_matrix(mat, figsize=(9, 9), colorbar=True)
    plt.show()

    print(get_f1_measure(4, predictions_SVM, y_test))


if __name__ == "__main__":
    main()
