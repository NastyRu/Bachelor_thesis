import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, \
                            QVBoxLayout, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QFont

WIDTH = 1000
HEIGHT = 700


class ImageLabel(QLabel):
    clicked = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setText('\n\n\n\n\n\n\n Перетащите изображение сюда или нажмите для выбора')
        self.setFont(QFont('Arial', 30))
        self.setStyleSheet('''
                            QLabel {
                                background-image: url("background.png");
                                color: rgb(168, 168, 168);
                                background-repeat: no-repeat;
                                background-position: center;
                                border: 2px dashed #aaa;
                            }
                            ''')

    def setPixmap(self, image):
        super().setPixmap(image)

    def mouseReleaseEvent(self, event):
        self.clicked.emit()


class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(WIDTH, HEIGHT)
        self.setAcceptDrops(True)
        self.setWindowTitle("Классификатор документов")

        layout = QVBoxLayout()

        self.photo_viewer = ImageLabel()
        layout.addWidget(self.photo_viewer)
        self.photo_viewer.clicked.connect(self.get_file)

        self.output = QLabel()
        self.output.setText('Ничего не загружено')
        self.output.setFont(QFont('Arial', 20))
        self.output.setFixedSize(1000, 25)
        self.output.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('''
                            QLabel {
                                color: rgb(64, 64, 64);
                            }
                            ы''')
        layout.addWidget(self.output)

        self.setLayout(layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        pixmap = self.get_pixmap(file_path)
        if pixmap:
            self.photo_viewer.setPixmap(pixmap)

    def get_file(self):
        file_path = QFileDialog.getOpenFileName(
            self, 'Open file',
            '/Users/anastasia/Desktop/Bachelor_thesis/src/examples',
            'Image files (*.jpg *.png *.jpeg)')
        self.set_image(file_path[0])

    def get_pixmap(self, file_path):
        if file_path == "":
            return None
        pixmap = QPixmap(file_path)
        return pixmap.scaled(WIDTH, HEIGHT, Qt.KeepAspectRatio)

    def update_label(self, path):
        self.output.setText('Ничего не загружено')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWidget()
    ui.show()
    sys.exit(app.exec_())
