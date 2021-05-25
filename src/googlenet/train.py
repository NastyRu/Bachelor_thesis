import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from inception_v1 import InceptionV1
import glob
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img


def make_train_data(class_type, class_dir, data, labels):
    for i in class_dir:
        image = load_img(i, color_mode='rgb', target_size=(224, 224))
        image = np.array(image)
        data.append(image)
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


def count_optimal_epochs(x_train, y_train,
                         x_test, y_test,
                         x_valid, y_valid,
                         epoch_steps, class_num):
    x = []
    y = []
    for epoch in range(1, 20):
        model = InceptionV1().architecture()
        optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=optimizer, metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=epoch,
                  steps_per_epoch=epoch_steps,
                  validation_data=(x_valid, y_valid))
        y_pred = model.predict(x_test)
        y_final = []
        for i in range(len(y_pred[2])):
            y_final.append(np.argmax(y_pred[2][i]))

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
            f1[i] = (f1_fn[i] + f1_fp[i]) \
                    / (2 * f1_tp[i] + f1_fn[i] + f1_fp[i])

        y.append(sum(f1) / len(f1))
        x.append(epoch)
    plt.plot(x, y)
    plt.show()


def main():
    data = []
    labels = []

    data, labels = make_train_data(0, glob.glob('training_data/passport/*.*'),
                                   data, labels)
    data, labels = make_train_data(1, glob.glob('training_data/passport1/*.*'),
                                   data, labels)
    data, labels = make_train_data(2, glob.glob('training_data/passport2/*.*'),
                                   data, labels)
    data, labels = make_train_data(3, glob.glob('training_data/passport3/*.*'),
                                   data, labels)
    data, labels = make_train_data(4, glob.glob('training_data/vu1/*.*'),
                                   data, labels)
    data, labels = make_train_data(5, glob.glob('training_data/vu2/*.*'),
                                   data, labels)
    data, labels = make_train_data(6, glob.glob('training_data/vu3/*.*'),
                                   data, labels)
    data, labels = make_train_data(7, glob.glob('training_data/visa_deu/*.*'),
                                   data, labels)
    data, labels = make_train_data(8, glob.glob('training_data/visa_esp/*.*'),
                                   data, labels)
    data, labels = make_train_data(9, glob.glob('training_data/visa_fra/*.*'),
                                   data, labels)
    data, labels = make_train_data(10, glob.glob('training_data/visa_ita/*.*'),
                                   data, labels)

    data = np.array(data)
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.4,
                                                        random_state=42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test,
                                                        test_size=0.5,
                                                        random_state=42)

    x_train = x_train / 255
    x_test = x_test / 255
    x_valid = x_valid / 255

    batch_size = 8
    epoch_steps = int(x_train.shape[0] / batch_size)
    model_name = 'googlenet_weights.h5'
    class_num = 11

    model = InceptionV1().architecture()
    model.summary()
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=12, steps_per_epoch=epoch_steps,
              validation_data=(x_valid, y_valid))

    model.save(model_name)

    y_pred = model.predict(x_test)
    y_final = []

    for i in range(len(y_pred[2])):
        y_final.append(np.argmax(y_pred[2][i]))

    mat = confusion_matrix(y_test, y_final)
    plot_confusion_matrix(mat, figsize=(9, 9), colorbar=True)
    plt.show()

    print(get_f1_measure(class_num, y_final, y_test))


if __name__ == "__main__":
    main()
