import sys
import os
import cv2

from PyQt5.uic.properties import QtGui

import united
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, pyqtSlot
from pathlib import Path
import shutil

import camera  # Это наш конвертированный файл дизайна


class CameraThread(QThread):
    frameCaptured = pyqtSignal(object)  # Сигнал для передачи кадра
    connectionError = pyqtSignal(str)  # Сигнал для передачи ошибки подключения

    # Открываем камеру
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        # self.cap = cv2.VideoCapture("rtsp://rsmc:ae4rut@192.168.120.127/1")
        self.running = True
        self.retry_interval = 10000  # Интервал времени для повторной попытки (в мс)

    def run(self):
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture("rtsp://rsmc:ae4rut@192.168.120.127/1")
                # self.cap = cv2.VideoCapture(0)
                # self.cap.set(cv2.CAP_PROP_FPS, 15)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if not self.cap.isOpened():
                    # В случае неправильного IP-адреса
                    self.connectionError.emit("Ошибка подключения. Перезагрузите приложение")
                    self.sleep(self.retry_interval // 10000)
                    continue

            ret, frame = self.cap.read()
            if ret:
                self.frameCaptured.emit(frame)
            else:
                # В случае потери интернет соединения
                self.connectionError.emit("Ошибка получения кадра")
                self.sleep(self.retry_interval // 10000)


    def stop(self):
        self.running = False
        self.cap.release()

class ExampleApp(QMainWindow, camera.Ui_MainWindow):
    def __init__(self):

        super().__init__()
        # Ui_MainWindow.setWindowTitle("Новое название окна")
        # Ui_MainWindow.setObjectName("MainWindow")
        self.camera_thread = CameraThread()
        self.camera_thread.frameCaptured.connect(self.update_frame)
        self.camera_thread.connectionError.connect(self.show_error)
        self.camera_thread.start()

        # self.textBrowser.setHtml("<p align='center'>")
        # self.textBrowser.setText("Ожидайте подключения камеры")

        # Создаем QLabel для отображения видео
        self.label_2 = QLabel(self)

        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        # self.textBrowser.setPlainText(" ")
        self.pushButton.clicked.connect(self.save_photo)  # Выполнить функцию save_photo при нажатии кнопки
        self.current_frame = None
        self.num = 0

        layout = QVBoxLayout()
        layout.addWidget(self.label_2)


        # self.label.setGeometry(0, 0, 1280, 720)

        # Инициализируем захват видео
        # self.cap = cv2.VideoCapture("rtsp://admin:admin@192.168.120.127/1")

        # Таймер для обновления кадров
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.cam_press)
        # self.timer.start(60)  # Обновление каждые 30 мс

    # Закоментить функцию
    # def cam_press(self):
    #     ret, frame = self.cap.read()
    #     frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    #     frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
    #     if ret:
    #         # Преобразование кадра в формат, понятный для PyQt
    #         rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         h, w, ch = rgbImage.shape
    #         bytesPerLine = ch * w
    #         qImg = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
    #         pixmap = QPixmap.fromImage(qImg)
    #         self.label_2.setPixmap(pixmap)

    def update_frame(self, frame):
        self.current_frame = frame
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # frame = cv2.resize(frame, (frame.shape[1] // 3, frame.shape[0] // 3))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        step = channel * width
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))

    def show_error(self, message):
        self.textBrowser.setHtml(f"<p align='center' style='color:red;'>{message}</p>")

    def save_photo(self):
        if self.current_frame is not None:
            # self.textBrowser.clear()
            # self.textBrowser.setText(" ")
            self.textBrowser.clear()
            self.label.clear()
            QApplication.processEvents()
            img_name = "opencv_frame.png"
            # project_dir = Path(__file__).parent
            # print(f"ДИРЕКТОРИЯ ЗАГРУЗКИ {project_dir}")
            # path_val = os.path.abspath('result/buffer')
            path_val = 'result/buffer'
            print(f"ДИРЕКТОРИЯ ЗАГРУЗКИ {path_val}")
            os.makedirs(path_val, exist_ok=True)
            self.current_frame = cv2.resize( self.current_frame, (1280, 720))
            cv2.imwrite(f'{path_val}/{img_name}', self.current_frame)
            print("Фото сохранено")
            path_val = 'seg'
            os.makedirs(path_val, exist_ok=True)
            cv2.imwrite(f'{path_val}/{img_name}', self.current_frame)
            print("Фото сохранено")

            path_weight = 'weights/model_12_43248_.pt'
            path_weight_seg = 'weights/model_12_batch_4_.pt'
            path_res = 'result'

            Text = '1'
            l = 0

            path_val = 'seg'
            num_test = self.add()

            Text = united.united(path_weight, path_weight_seg, path_val, path_res, num_test)
            # self.textBrowser.setText(str(Text))

            path_val = 'result/buffer'
            # В зависимости от того верно или нет у нас создается папка
            # Err - если не правильно, ok -  правильно
            if Text == 'Полярность верная':
                result_dir = f'result/ok{num_test}'
                self.textBrowser.setHtml(f"<p align='center' style='color:green;'>{Text}</p>")

            else:
                result_dir = f'result/err{num_test}'
                self.textBrowser.setHtml(f"<p align='center' style='color:red;'>{Text}</p>")

            os.makedirs(result_dir, exist_ok=True)
            for file in os.listdir(path_val):
                # Полный путь к файлу
                file_path = os.path.join(path_val, file)
                # Проверяем, что это файл и имеет нужное расширение
                if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    # Копируем файл в папку назначения
                    shutil.copy(file_path, os.path.join(result_dir, file))

            # Удалим все файлы из папки Buffer, чтобы они не перемешивались в случае ошибок с другими фотографиями
            # for file in os.listdir(path_val):
            #     file_path = os.path.join(path_val, file)
            #     try:
            #         if os.path.isfile(file_path) or os.path.islink(file_path):
            #             os.unlink(file_path)


            path_to_photo = 'result/buffer/QR.png'
            # path_to_photo = result_dir / 'QR.png'
            frame = cv2.imread(str(path_to_photo))
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            qImg = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.label.setPixmap(pixmap)



    def closeEvent(self, event):
        self.camera_thread.stop()
        self.camera_thread.wait()
        event.accept()


    def add(self):
        self.num += 1
        return self.num

# try:

def main():
    app = QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    # app.exec_()  # Запускает бесконечный цикл
    sys.exit(app.exec_())

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
