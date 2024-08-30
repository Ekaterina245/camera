import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def calibration():

    # Определение количества кввадратов на шахматной доскеи
    CHECKERBOARD = (6, 9)
    # Переменная, задающая критерии остановки алгоритма оптимизации (останавливается, когда выполнено одно из условий)
    # cv2.TERM_CRITERIA_EPS - требуемая точность 0.001
    # cv2.TERM_CRITERIA_MAX_ITER - кол-во итераций (30)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Создание вектора для хранения векторов трехмерных точек для каждого изображения шахматной доски
    objpoints = []
    # Создание вектора для хранения векторов 2D точек для каждого изображения шахматной доски
    imgpoints = []

    # Определение мировых координат для 3D точек
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Извлечение пути отдельного изображения, хранящегося в данном каталоге

    # project_dir = Path(__file__).parent
    # images = list(project_dir.glob('distor2/*.png'))
    # print(type(images), images)
    path_val = 'distor2'
    images = os.listdir(path_val)
    print(type(images), images)
    list_all_name = []
    # print(type(list_all_name))
    for name in images:
        name_pic = os.path.join(path_val, name)
        # print(f'NAME_PIC {name_pic}, {type(name_pic)}')
        list_all_name.append(name_pic)
    # print(f'list_all_name {list_all_name}')

    gray = 0
    img = 0
    for fname in list_all_name:
        # print("NAME", fname)
        img = cv2.imread(fname)
        # print(f'GRAY_shape {img.shape}')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(f'GRAY_shape {gray.shape}')
        # Найти углы шахматной доски
        # Если на изображении найдено нужное количество углов, тогда ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        Если желаемый номер угла обнаружен,
        уточняем координаты пикселей и отображаем
        их на изображениях шахматной доски
        """
        if ret == True:
            objpoints.append(objp)
            # уточнение координат пикселей для заданных 2d точек.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Нарисовать и отобразить углы
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # print(img.shape)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    # h, w = img.shape[:2]
    # print(f'H, W {h}, {w}')

    """
    Выполнение калибровки камеры с помощью
    Передача значения известных трехмерных точек (объектов)
    и соответствующие пиксельные координаты
    обнаруженные углы (imgpoints)
    """
    # print(f'ERROR {gray.shape[::-1]}')
    # print(f'TYPE {type(gray)}')
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return ret, mtx, dist, rvecs, tvecs

def calibration_for_foto(path_val):
    ret, mtx, dist, rvecs, tvecs = calibration()

    imgs = glob.glob(f'{path_val}/*.png')
    # print("ДЛИНА СПИСКА", len(imgs))
    # print(f'Сам список фотографий {imgs[0]}')
    for i in range(0, len(imgs)):
        img = cv2.imread(imgs[i])[...,::-1]
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        h, w = img.shape[:2]
        # Уточнение матрицы для иправления дисторсии
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        # Переписываем изображение с учетом поправочных коофициентов
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # print(f'Размерность фотографии {dst.shape}')
        # Обрезаем изображение
        # x, y, w, h = roi
        # print(f'Результаты {x, y, w, h}')
        # dst = dst[y:y + h, x:x + w]
        # print(f'Размерность новой фотографии {dst.shape}')

        # Сохраняем фотографию вместо сделанной с камеры
        # Проверить одинаковые ли имена у фото с камеры и сохраненного
        # убрала в конце {i} чтобы фото перезаписывалась
        filename2 = f'opencv_frame.png'
        im = Image.fromarray(dst)
        full_path = os.path.join(path_val, filename2)
        im.save(full_path)

# project_dir = Path(__file__).parent
# calibration_for_foto(path_val=project_dir / 'seg')