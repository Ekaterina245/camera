import cv2
import matplotlib.pyplot as plt

import numpy as np
import imutils
from imutils import contours
from skimage import measure

import skimage
from skimage.measure import label, regionprops
from skimage.draw import rectangle_perimeter

import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.morphology import closing, square
from skimage.color import label2rgb

import math
import os

import CNN
from sys import argv
from pathlib import Path
# import harris_corners

# Функция определяющая пятна синего цвета
# result - список с координатами рамок, выделяющих эти объекты, площадь которых находится в диапазоне от 240 до 3000 пикселей
# thresh - изображение после дилотации, двумерное изображдение в чб
# Используется для проверки, что 12 сегментов в блоке, обнаружения сначала синего на оригинальном изображении,
# затем для обнаружения на повернутой фотографии, для перепроверки, что мы не обрезали часть акума во время поворота и
# для нахождения координат расположиния кодов справа и слева
def function_for_transform(image):
    # Разделение изображение на 3 канала RGB
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]

    # Отображение трех каналов для сравнения яркости
    # fig = plt.figure(figsize = (10, 20))
    # row = 3
    # col = 1
    #
    # fig.add_subplot(row, col, 1)
    # plt.imshow(red, cmap = 'gray')
    # plt.title("RED")
    #
    # fig.add_subplot(row, col, 2)
    # plt.imshow(green, cmap = 'gray')
    # plt.title("GREEN")
    #
    # fig.add_subplot(row, col, 3)
    # plt.imshow(green, cmap = 'gray')
    # plt.title("BLUE")
    # plt.show()
    # plt.imshow(blue, cmap='gray')
    # plt.title('Синий канал')
    # plt.show()

    green_2 = np.where((green >=155) & (green<=227), 1, 0)
    # plt.imshow(green_2, cmap='gray')
    # plt.title('Преобразованный зеленый 1')
    # plt.show()

    red_2 = np.where((red >= 10) & (red <= 88), 1, 0)
    # plt.imshow(red_2, cmap='gray')
    # plt.title('Преобразованный красный 1')
    # plt.show()

    # Выбор интересующих пикселей при помощи порога синего
    L_limit = np.array([1, 149, 231])
    U_limit = np.array([93, 238, 255])
    b_mask = cv2.inRange(image, L_limit, U_limit)
    blue = cv2.bitwise_and(blue, b_mask)
    plt.imshow(blue, cmap='gray')
    plt.title('Синий канал')
    plt.show()

    # Не смотря на то, что массивы одного формата, они не объединяются,
    # поэтому приводим к одному типу и формату
    if blue.shape != red_2.shape:
        red_2 = cv2.resize(red_2, (blue.shape[1], red_2.shape[0]))

    if blue.dtype != red_2.dtype:
        red_2 = red_2.astype(blue.dtype)

    # Объединение красного канала и выделенного голубого цвета
    b_r = cv2.bitwise_or(blue, red_2)
    # Переменная, чтобы записать результат объединения, удобно менять, если объединили другие каналы
    chanel = b_r

    # Размытие по Гаусу
    blurred = cv2.GaussianBlur(chanel, (11, 11), 0)
    plt.imshow(blurred, cmap='gray')
    plt.title('Размытие по Гаусу')
    plt.show()

    # Порог яркости
    thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(thresh, cmap = 'gray')
    # plt.title("Изображение, прощедшее порог яркости")
    # plt.show()

    # Эрозия и дилотация
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    # plt.imshow(thresh, cmap = 'gray')
    # plt.title('Эрозия и дилотация')
    # plt.show()

    # Нахождение объектов на изображении
    labeled_image = label(thresh)
    # print(label(thresh, connectivity=1))

    # Получение свойств областей
    regions = regionprops(labeled_image)
    image_label_overlay = label2rgb(labeled_image, image=image, bg_label=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    result = []
    i = 0
    for region in regions:
        # print(region.bbox)
        # Координаты ограничивающего прямоугольника
        minr, minc, maxr, maxc = region.bbox
        print(f"Region with label {region.label}: Bounding box coordinates: ({minr}, {minc}) to ({maxr}, {maxc})")
        square = abs((maxr - minr) * (maxc - minc))
        print(f"Region with label {region.label}, {square}")

        if square >= 290 and square <= 3300:
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor='red',
                linewidth=2,
            )
            ax.add_patch(rect)

            result.append([minr, minc, maxr, maxc])
        # i += 1

    plt.tight_layout()
    plt.title("Обнаружение синих овалов")
    plt.show()

    return result, thresh

# Функция в которой собрана первичная обработка изображения, размытие по Гаусу, порог яркости, эрозия и дилатация
# Возвращается изображение типа и размера: <class 'numpy.ndarray'> (1083, 55)
def Gauss_blur_erode(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (11, 11), 0)
    # plt.imshow(blurred, cmap='gray')
    # plt.title('Размытие по Гаусу')
    # plt.show()

    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    # plt.imshow(thresh, cmap='gray')
    # plt.title("Изображение, прощедшее порог яркости")
    # plt.show()

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    # plt.imshow(thresh, cmap='gray')
    # plt.title('Эрозия и дилотация')
    # plt.show()

    return thresh

# Нахождение кодов на изображении. Вывод координат и их обрисовка ограничивающими рамками
def visual (right_trans, image, num_test):

    # Нахождение объектов на изображении
    labeled_image = label(right_trans)

    # Получение свойств областей
    regions = regionprops(labeled_image)
    image_label_overlay = label2rgb(labeled_image, image=image, bg_label=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)

    # Вывод координат и обрисовка белых областей
    result_r = []
    result_end_r = []
    for region in regions:
        # print(region.bbox)
        # Координаты ограничивающего прямоугольника
        minr, minc, maxr, maxc = region.bbox
        print(f"Region with label {region.label}: Bounding box coordinates: ({minr}, {minc}) to ({maxr}, {maxc})")
        square = abs((maxr - minr) * (maxc - minc))
        print(f"Region with label {region.label}, {square}")

        if square >= 200 and square <= 1200:
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor='red',
                linewidth=2,
            )
            ax.add_patch(rect)
            result_r.append([minr, minc, maxr, maxc])
            cv2.rectangle(image_label_overlay, (minc, minr), (maxc, maxr), color=(255, 0, 0), thickness=2)


        elif square > 1200:
            rect = mpatches.Rectangle(
                (minc, minr),
                maxc - minc,
                maxr - minr,
                fill=False,
                edgecolor='red',
                linewidth=2,
            )
            ax.add_patch(rect)
            result_end_r.append([minr, minc, maxr, maxc])


    # project_dir = Path(__file__).parent
    result_dir = 'result/buffer/'
    result_dir = str(result_dir)
    filename = 'QR'

    # print("КОДЫ", result_r)
    # print("Большие объекты", result_end_r)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(result_dir + filename, bbox_inches='tight', pad_inches=0)
    plt.show()

    # project_dir = Path(__file__).parent
    # result_dir = 'result/buffer'
    # result_dir = str(result_dir)
    # filename = 'QR'
    # plt.figure(figsize=(20, 30))
    # # plt.title(f"Другой способ")
    # image_label_overlay = cv2.rotate(image_label_overlay, cv2.ROTATE_90_CLOCKWISE)
    # print(f'SHAPE {image_label_overlay.shape}, ERROR {image_label_overlay}')
    # print(f"Минимальное значение: {np.min(image_label_overlay)}")
    # print(f"Максимальное значение: {np.max(image_label_overlay)}")
    # # image_label_overlay = np.where(image_label_overlay >= 0.55, 0, 1)
    # # image_label_overlay = image_label_overlay/255.0
    # plt.imshow(image_label_overlay)
    # plt.savefig(result_dir + filename, bbox_inches='tight', pad_inches=0)
    # plt.show()

    return result_r, result_end_r


def main(path_img, x1, y1, x2, y2, mask_after, num_test):
    # Загружаем оригинальное изображение
    img_path_cnn = 0
    imgs_cnn = list(sorted(os.listdir(path_img)))
    # print("ДЛИНА СПИСКА", len(imgs_cnn))
    for i in range(0, len(imgs_cnn)):
        img_path_cnn = os.path.join(path_img, imgs_cnn[i])

    image = cv2.imread(img_path_cnn)[...,::-1]

    # Обрезаем только интересующую часть ко координатам рамки предсказаннной нейронкой
    # Координаты возвращаются из скрипта нейронки
    image = image[y1:y2, x1:x2]
    # plt.imshow(image)
    # plt.title('Обрезанное  изображение')
    # plt.show()

    # Вызываем фуункцию для обнаружения ярких пятен
    result, thresh = function_for_transform(image)
    # print(f'КОЛИЧЕСТВО ОБНАРУЖЕННЫХ ОВАЛОВ', len(result))

    # TextQT = ''
    # l = 0
    arct_foto = 0

    # Расчет наклона фотографии
    num = (result[-1][0] - result[0][0]) / (result[-1][1] - result[0][1])
    arct_foto = abs(180 * math.atan(num) / math.pi)
    print(f'УГОЛ НАКЛОНА ФОТОГРАФИИ {arct_foto}')
    # print("SHAPE", image.shape)
    height, width = thresh.shape
    # print(height, width)
    rotate_foto = cv2.getRotationMatrix2D((width / 2, height / 2), arct_foto, 1)
    rotate_foto = cv2.warpAffine(image, rotate_foto, (width, height))

    # fig = plt.figure(figsize=(10, 6))
    # row = 1
    # colomns = 2
    #
    # fig.add_subplot(row, colomns, 1)
    # plt.title("Original")
    # plt.imshow(image, cmap='gray')
    #
    # fig.add_subplot(row, colomns, 2)
    # plt.title("Rotate")
    # plt.imshow(rotate_foto, cmap='gray')
    # plt.show()

    # Координаты новых повернутых овалов
    # print(f'Размерность повернутой фотографии - {rotate_foto.shape}')
    result_rotate = []
    result_rotate, thresh_rotate = function_for_transform(rotate_foto)

    # plt.imshow(thresh_rotate, cmap='gray')
    # plt.title("После разворота")
    # plt.show()

    # Расчет координат для обрезки области возле QR-кодов
    height, width = thresh_rotate.shape

    # Формат result_rotate 0 - y1, 1 - x1, 2 - y2, 3 - x2
    print(f'КООРДИНАТЫ ПОСЛЕ РАЗВОРОТА ПО КОТОРЫМ СЧИТАЕМ {result_rotate}')
    # Для вертикального фото
    # x_center = int((result_rotate[0][3] - result_rotate[0][1]) / 2 + result_rotate[0][1])
    # Для горизонтального фото
    у_center = int((result_rotate[0][2] - result_rotate[0][0]) / 2) + result_rotate[0][0]
    print(f'координаты центра - {у_center}')

    # горизонтальная область обрезки

    shift = 50
    y_cen_down = int(у_center - shift)
    # print(f'координаты down - {y_cen_down}')
    y_cen_up = int(у_center + shift)
    # print(f'координаты up - {y_cen_up}')

    up_down = rotate_foto[y_cen_down:y_cen_up, 0: width]
    # print(f'Crop - {up_down}, {up_down.shape}')

    # Преобразуем в оттенки серого
    # right_img = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    # left_img = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    # up_down_img = cv2.cvtColor(up_down, cv2.COLOR_BGR2GRAY)

    # На данный момент лучше себя показывает разделение на каналы, и выбор красного, так как в нем кода самые светлые
    up_down_img_red = up_down[:, :, 0]
    up_down_img_green = up_down[:, :, 1]
    up_down_img_blue = up_down[:, :, 2]

    # plt.imshow(up_down_img_red, cmap='gray')
    # plt.title("Центр изображения с кодами")
    # plt.show()

    #######
    # Часть с выделением кодов
    # right_trans = Gauss_blur_erode(right_img)
    # left_trans = Gauss_blur_erode(left_img)
    center_trans = Gauss_blur_erode(up_down_img_red)
    # print("RIGHT_TRANS", right_trans.shape)

    # Центральная часть
    # Список для хранения координат всех рамок (result_c), будем сравнивать их крайнюю координату с растоянием до угла
    # Записываем в список и считаем по по первым и последним 6 элементам
    result_с = []
    result_end_с = []
    result_с, result_end_с = visual(center_trans, up_down_img_red, num_test)
    print("Итоговый список", result_с)
    print("Первый Х", result_с[0][1])

    # new_list_distance = []
    # for i in range(len(result_с)):
    #     new_list_distance.append(result_с[i][2])
    # print(f'Координаты нижних краёв рамок {new_list_distance}')

    result_с = sorted(result_с, key =lambda x:x[1])
    print("SORTED",result_с)
    # percent = []
    # for i in range(len(result_с)):
    #     percent.append((new_list_distance[i] / (shift*2)) * 100)
    # print(f'Список с процентным содержанием {percent}, {len(percent)}')

    height_QR = 100
    weights_OR = 1280
    # Проверка процентного нахождения
    fl = True
    if len(result_с) == 12:

        for i, result_box in enumerate(result_с):
            # Размеры бокса в нормальном виде
            cur_reg = result_box[:]
            x = cur_reg[1]
            y = cur_reg[0]
            w = cur_reg[3] - cur_reg[1]
            h = cur_reg[2] - cur_reg[0]

            if (i+1)%2 == 1:
                distance = height_QR - (y+h) #100 - (75+14)
            else:
                distance = height_QR - (height_QR-y)
            print(f"DISTANCE {distance}")

            if distance * 2 > height_QR:
                fl = False
                TextQT = 'Полярность неверная'
                print("Полярность неверная")
                break

        if fl == True:
            TextQT = 'Полярность верная'
            print("Полярность верная")

    else:
        l = 0
        TextQT = 'Полярность неверная'
        print("Полярность неверная")

    # if l == 1:
    #     TextQT = 'Полярность верная'
    #     print("Полярность верная")


    return TextQT


# if __name__ == "__main__":
#     # script, path_weight, path_val, path_res = argv
#     project_dir = Path(__file__).parent
#     path_weight = project_dir / 'weights' / 'model_12_43248_.pt/'
#     path_weight_seg = project_dir / 'weights' / 'model_12_batch_4_.pt'
#     path_val = project_dir / 'seg'
#     # Для сохранения картинки для полного отчета об одном заупске
#     path_res = project_dir / 'result'
#     num_test = 3
#     number, x1, y1, x2, y2, mask_after = CNN.CNN(path_weight, path_val, num_test, 'all')
#     print("ИТОГОВЫЕ КООРДИНАТЫ", x1, y1, x2, y2, mask_after.shape)
#     text = main(path_val, x1, y1, x2, y2, mask_after, num_test)
#     print("Значения ТЕКСТА И ФЛАГА", text)


