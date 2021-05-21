import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD
from googlenet.inception_v1 import InceptionV1
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


def main():
    data = []
    labels = []

    data, labels = make_train_data(0, glob.glob('content/classes/class0/*.*'),
                                   data, labels)
    data, labels = make_train_data(1, glob.glob('content/classes/class1/*.*'),
                                   data, labels)
    data, labels = make_train_data(2, glob.glob('content/classes/class2/*.*'),
                                   data, labels)
    data, labels = make_train_data(3, glob.glob('content/classes/class3/*.*'),
                                   data, labels)
    data, labels = make_train_data(4, glob.glob('content/classes/class4/*.*'),
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

    batch_size = 16
    epoch_steps = int(x_train.shape[0] / batch_size)
    model_name = 'googlenet_weights.h5'

    model = InceptionV1().architecture()
    model.summary()

    # optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=optimizer,
    #               metrics=['accuracy'])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20, steps_per_epoch=epoch_steps,
              validation_data=(x_valid, y_valid))

    model.save(model_name)

    y_pred = model.predict(x_test)
    y_final = []

    for i in range(len(y_pred[2])):
        y_final.append(np.argmax(y_pred[2][i]))

    mat = confusion_matrix(y_test, y_final)
    plot_confusion_matrix(mat, figsize=(9, 9), colorbar=True)
    plt.show()


if __name__ == "__main__":
    main()
