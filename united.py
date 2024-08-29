import matplotlib.pyplot as plt
import cv2
import camera_foto
import calibration
import CNN
from sys import argv
import blue_object
import time
import os
from pathlib import Path
import shutil

# script, path_weight, path_val, path_res = argv

def united(path_weight, path_weight_seg, path_val, path_res, num_test):
    # Пути к весам, папке с тестовыми фото, к папке, куда надо сохранить результат
    # script, path_weight, path_val, path_res = argv
    # flag_cam = 0
    # flag_cam = camera_foto.main(path_val, path_res)
    # print(flag_cam)
    flag_cam = 1
    # Запуск дальнейшей программы только после того как camera.py закончит работать
    if flag_cam == 1:
        # Калибровка полученных фотографий
        # calibration.calibration_for_foto(path_val)
        # Координаты предсказанной рамки
        number, x1, y1, x2, y2, mask_after = CNN.CNN(path_weight, path_val, num_test, 'all')
        print("ИТОГОВЫЕ КООРДИНАТЫ", x1, y1, x2, y2, mask_after.shape)

        # Подсчет сегментов
        number_seg, x3, y3, x4, y4, mask_after_seg = CNN.CNN(path_weight_seg, path_val, num_test, 'every')
        if number_seg == 12:
            # Код openCV, определяет количество сегментов и правильную полярность
            Text = blue_object.main(path_val, x1, y1, x2, y2, mask_after, num_test)
            print("Значения ТЕКСТА И ФЛАГА", Text)
        else:
            Text = 'Неверное количество сегментов'
            print("Значения ТЕКСТА И ФЛАГА", Text)
            # project_dir = Path(__file__).parent
            # Откуда сохраняем
            file_name_er = 'error/QR.png'
            # Куда сохраняем
            result_dir = 'result/buffer'
            # Проверяем, существует ли целевая папка, и если нет, создаем ее
            os.makedirs(result_dir, exist_ok=True)

            # Копируем файл в целевую папку
            shutil.copy(file_name_er, result_dir)

            # # Проходимся по вссем файлам в папке error
            # # for file in sourse_dir.iterdir():
            # #     # Проверяем, что файл является изображением (например, с расширением .jpg, .png)
            # #     if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
            # #         # Копируем файл в новую папку
            # #         shutil.copy(file, result_dir / file.name)
            #
            # for file in os.listdir(sourse_dir):
            #     # Полный путь к файлу
            #     file_path = os.path.join(sourse_dir, file)
            #     # Проверяем, что это файл и имеет нужное расширение
            #     if os.path.isfile(file_path) and file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
            #         # Копируем файл в папку назначения
            #         shutil.copy(file_path, os.path.join(sourse_dir, file))

    return Text

# project_dir = Path(__file__).parent
# path_weight = project_dir / 'weights' /'model_12_43248_.pt/'
# path_weight_seg = project_dir / 'weights' / 'model_12_batch_4_.pt'
# path_val = project_dir / 'seg'
# #Для сохранения картинки для полного отчета об одном заупске
# path_res = project_dir / 'result'
# Text = '1'
# l = 0
# united(path_weight, path_weight_seg, path_val, path_res, 3)






