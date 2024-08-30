import sys
import os
import cv2
import time
from PyQt5.uic.properties import QtGui

import united
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, pyqtSlot
from pathlib import Path
import shutil

import camera  # Это наш конвертированный файл дизайна
import asyncio
import nest_asyncio
from aiortc.contrib.media import MediaPlayer

from datetime import date

current_date = date.today()
print("Текущая дата:", current_date)

nest_asyncio.apply()

class CameraThread(QThread):
    frameCaptured = pyqtSignal(object)  # Сигнал для передачи кадра
    connectionError = pyqtSignal(str)  # Сигнал для передачи ошибки подключения

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.retry_interval = 10  # Интервал времени для повторной попытки (в секундах)
        with open("IP.txt") as f:
            path = f.readline()
        self.rtsp_url = path
        self.target_width = 1280
        self.target_height = 720

    async def async_run(self):
        while self.running:
            try:
                start = time.time()
                player = MediaPlayer(self.rtsp_url, options={"rtsp_transport": "tcp", "buffer_size": "10000000"}, timeout=10)
                track = player.video
                print(type(track))
                if not player or not track:
                    raise ConnectionError("Не удалось подключиться к RTSP потоку")

                while self.running:
                    try:
                        frame = await track.recv()
                        # Проверяем, если кадр пустой
                        if frame is None:
                            raise ConnectionError("Потеряно соединение с RTSP потоком")

                        # Конвертируем кадр в numpy массив (формат BGR)
                        img = frame.to_ndarray(format="bgr24")

                        # Приведение изображения к нужному размеру
                        resized_img = cv2.resize(img, (self.target_width, self.target_height))

                        # Эмитируем сигнал с кадром
                        self.frameCaptured.emit(resized_img)

                        stop = time.time()
                        # print('Результирующее время запуска камеры', stop - start)

                    except Exception as e:
                        # Обработка возможных ошибок при получении кадров
                        self.connectionError.emit(f"Ошибка получения кадра")
                        await asyncio.sleep(self.retry_interval)
                        break  # Выход из внутреннего цикла, чтобы попытаться переподключиться

                    # Даем возможность обработки других событий в Qt
                    await asyncio.sleep(0.01)


            except Exception as e:
                # В случае ошибки подключения
                self.connectionError.emit(f"Ошибка подключения")
                await asyncio.sleep(self.retry_interval)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.async_run())

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class ExampleApp(QMainWindow, camera.Ui_MainWindow):
    def __init__(self):

        super().__init__()
        self.camera_thread = CameraThread()
        self.camera_thread.frameCaptured.connect(self.update_frame)
        self.camera_thread.connectionError.connect(self.show_error)
        self.camera_thread.start()

        # Создаем QLabel для отображения видео
        self.label_2 = QLabel(self)

        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        # self.textBrowser.setPlainText(" ")
        self.pushButton.clicked.connect(self.save_photo)  # Выполнить функцию save_photo при нажатии кнопки

        self.current_frame = None
        self.num = 0

        layout = QVBoxLayout()
        layout.addWidget(self.label_2)

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
            time_foto = time.time()
            self.textBrowser.clear()
            self.label.clear()
            QApplication.processEvents()
            img_name = "opencv_frame.png"
            # print(f"ДИРЕКТОРИЯ ЗАГРУЗКИ {project_dir}")
            path_val = 'result/buffer'
            print(f"ДИРЕКТОРИЯ ЗАГРУЗКИ {path_val}")
            os.makedirs(path_val, exist_ok=True)
            self.current_frame = cv2.resize( self.current_frame, (1280, 720))
            cv2.imwrite(f'{path_val}/{img_name}', self.current_frame)
            print("Фото сохранено")
            time_foto_after = time.time()
            time_after = time_foto_after - time_foto
            print(f'FOTO_AFTER_BUFFER, {time_after:.5f}')
            path_val = 'seg'
            os.makedirs(path_val, exist_ok=True)
            cv2.imwrite(f'{path_val}/{img_name}', self.current_frame)
            print("Фото сохранено")

            # path_weight = 'weights/model_12_43248_.pt'
            path_weight = 'weights/finally_model_for_all.pt'
            # path_weight_seg = 'weights/model_12_batch_4_.pt'
            path_weight_seg = 'weights/finally_model_for_every.pt'
            path_res = 'result'

            Text = '1'
            l = 0

            path_val = 'seg'
            num_test = self.add()

            Text = united.united(path_weight, path_weight_seg, path_val, path_res, num_test)
            # self.textBrowser.setText(str(Text))
            # path_val = f'result/{current_date}/buffer'
            path_val = 'result/buffer'
            # В зависимости от того верно или нет у нас создается папка
            # Err - если не правильно, ok -  правильно
            if Text == 'Полярность верная':
                result_dir = f'result/{current_date}/ok{num_test}'
                self.textBrowser.setHtml(f"<p align='center' style='color:green;'>{Text}</p>")

            else:
                result_dir = f'result/{current_date}/err{num_test}'
                self.textBrowser.setHtml(f"<p align='center' style='color:red;'>{Text}</p>")


            print(f'time__before_buff_err: {time.time():.5f}')

            os.makedirs(result_dir, exist_ok=True)
            for file in os.listdir(path_val):
                # Полный путь к файлу
                file_path = os.path.join(path_val, file)
                # Проверяем, что это файл и имеет нужное расширение
                if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                    # Копируем файл в папку назначения
                    print(f'time_1_pic: {time.time():.5f}')
                    shutil.copy(file_path, os.path.join(result_dir, file))


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

            # Удалим все файлы из папки Buffer, чтобы они не перемешивались в случае ошибок с другими фотографиями
            for file in os.listdir(path_val):
                file_path = os.path.join(path_val, file)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Не удалось удалить {file_path}')



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
    # Запускает бесконечный цикл
    sys.exit(app.exec_())


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()
