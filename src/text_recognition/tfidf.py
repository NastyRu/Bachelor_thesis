from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import os
import numpy as np
import glob


# Перевод текста в нижний регистр
def convert_to_lowercase(data):
    return np.char.lower(data)


# Удаление стоп-слов
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = str(data).split('\n')
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


# Удаление пунктуации
def remove_punctuation(data):
    symbols = "!#$%&()*+./:;=?@[]^_`{|}~"
    for sym in symbols:
        data = np.char.replace(data, sym, " ")
        data = np.char.replace(data, "  ", " ")

    return data


# Удаление суффисов и окончаний
def stemming(data):
    stemmer = PorterStemmer()

    tokens = str(data).split('\n')
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


# Препроцессинг текста
def preprocess(data):
    data = convert_to_lowercase(data)
    data = remove_punctuation(data)
    return data


def doc_freq(word, DF):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c


def ifIn(str, words):
    iss = False
    for word in words:
        if (word.find(str) != -1):
            print(word, str)
            iss = True
    return iss


# TF-IDF matching score ranking
def matching_score(k, query, tf_idf):
    preprocessed_query = preprocess(query)
    tokens = str(preprocessed_query).split('\n')

    query_weights = {}

    for key in tf_idf:
        if ifIn(key[1], tokens):
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]

    print(query_weights)
    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)
    print(query_weights)

    l = []

    for i in query_weights[:10]:
        l.append(i[0])
    return l


# Извлечение текста из файлов
def read_files(files):
    processed_text = []

    for file in files:
        f = open(file, 'r', encoding='utf8', errors='ignore')
        text = f.read().strip()
        f.close()
        processed_text.append(str(preprocess(text)).split('\n'))

    return processed_text


# Оценка важности слова в документе
def get_tfidf(files):
    processed_text = read_files(files)

    DF = {}
    N = len(files)
    for i in range(N):
        tokens = processed_text[i]
        for token in tokens:
            try:
                DF[token].add[i]
            except:
                DF[token] = {i}
    for i in DF:
        DF[i] = len(DF[i])
    print(DF)

    doc = 0
    tf_idf = {}
    print(N)

    for i in range(N):
        tokens = processed_text[i]
        counter = Counter(tokens)
        words_count = len(tokens)

        for token in np.unique(tokens):
            tf = counter[token]/words_count
            df = doc_freq(token, DF)
            idf = np.log((N+1)/(df+1))
            tf_idf[doc, token] = tf * idf
        doc += 1

    for i in tf_idf:
        tf_idf[i] *= 0.3

    print(tf_idf)

    return tf_idf


def main():
    files = glob.glob("documents_keyword/*.txt")
    tf_idf = get_tfidf(files)

    test_file_path = os.getcwd()+'/downloads'+'/file_pnrus.txt'
    f = open(test_file_path, 'r', encoding='utf8', errors='ignore')
    query = f.read().strip()
    f.close()

    l = matching_score(2, query, tf_idf)
    print(l)


    for i in range(len(l)):
        print(l[i], files[l[i]])


if __name__ == "__main__":
    main()
