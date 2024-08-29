# Часть необходимая чтобы не создавалось несколько потоков и процессор не перегружался
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Импорт файлов для доп. функций
# from load_dataset import get_prediction, segment_instance
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
import random

import torchvision.transforms as T
import numpy as np
import warnings

from sys import argv

warnings.filterwarnings('ignore')

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision
import numpy as np

import cv2
import random
import warnings

from pathlib import Path

warnings.filterwarnings('ignore')


def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 255], [255, 255, 0], [255, 0, 255], [80, 70, 180],
               [250, 80, 190], [245, 145, 50], [70, 150, 250], [50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0, 10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(img_path, confidence, model, all_or_every):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    # print(img)

    CLASS_NAMES = ['__background__', 'battery']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    img = img.to(device)
    pred = model([img])
    # print(pred)
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]

    max_confidence = max(pred_score)

    if all_or_every == "every":
        pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]
    else:
        pred_t = [pred_score.index(x) for x in pred_score if x == max_confidence][-1]

    # print((pred[0]['masks'] > 0.1).squeeze().detach().cpu().numpy())
    # print((pred[0]['masks'] > 0.2).squeeze().detach().cpu().numpy())
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    # print(pred[0]['labels'].numpy().max())
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t + 1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_class = pred_class[:pred_t + 1]
    return masks, pred_boxes, pred_class, max_confidence


def segment_instance(img_path, path_weight, num_test, filename, all_or_every, confidence=0.2, rect_th=2, text_size=2, text_th=2, key=0):
    """
    segment_instance
      Параметры:
        - img_path - путь к папке с проверяемыми изображениями
        - confidence- уверенность в предсказании от 0 до 1
        - rect_th - толшина линий прямоугольника
        - text_size - размер текста
        - text_th - толщина текста
      Метод:
        - Предсказание берется из get_prediction
        - Каждое изображение считывается при помощи opencv
        - Каждой маске дается случайный цвет функцией get_coloured_mask
        - each mask is added to the image in the ration 1:0.8 with opencv

    """

    model = torch.load(path_weight, map_location=torch.device('cpu'))
    model.eval()
    CLASS_NAMES = ['__background__', 'battery']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    # project_dir = Path(__file__).parent
    result_dir = 'result/buffer'
    result_dir = str(result_dir)
    number_of_seg = 1
    x, y, x_1, y_1 = 0, 0, 0, 0
    if all_or_every == "every":
        masks, boxes, pred_cls, max_confidence = get_prediction(img_path, confidence, model, all_or_every)
        count_masks = len(masks)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            # rgb_mask = get_coloured_mask(masks[i])
            # img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            boxes[i][0] = list(boxes[i][0])
            boxes[i][1] = list(boxes[i][1])
            for j in range(2):
                for u in range(2):
                    boxes[i][j][u] = int(boxes[i][j][u])
            boxes[i][0] = tuple(boxes[i][0])
            boxes[i][1] = tuple(boxes[i][1])
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)

        plt.figure(figsize=(20, 30))
        plt.title(f"Результат с точностью {confidence},       Количество масок {count_masks}")
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(result_dir + filename, bbox_inches='tight', pad_inches=0)
        plt.show()
        return len(masks), x, y, x_1, y_1, masks

    else:
        masks, boxes, pred_cls, max_confidence = get_prediction(img_path, confidence, model, all_or_every)
        count_masks = len(masks)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            # rgb_mask = get_coloured_mask(masks[i])
            # img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
            boxes[i][0] = list(boxes[i][0])
            boxes[i][1] = list(boxes[i][1])
            for j in range(2):
                for u in range(2):
                    boxes[i][j][u] = int(boxes[i][j][u])
            boxes[i][0] = tuple(boxes[i][0])
            boxes[i][1] = tuple(boxes[i][1])
            cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0),
                        thickness=text_th)
            # filename = f'C:/Users/EVM/Documents/camera_test/result/result_only_test/29/img_{random.randint(1, 100)}.png'

            koor1, koor2 = boxes[0]
            x, y = koor1
            x_1, y_1 = koor2
            key = 1
            print(f'TYPE {type(result_dir)}, {type(filename)}')
            plt.figure(figsize=(20, 30))
            plt.title(f"Результат с точностью {max_confidence}")
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(result_dir + filename, bbox_inches='tight', pad_inches=0)
            plt.show()

            # img = cv2.imread(result_dir + filename)
            # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # plt.imshow(img)
            # plt.savefig(result_dir + filename, bbox_inches='tight', pad_inches=0)
            # plt.show()
            return max_confidence, x, y, x_1, y_1, masks


def CNN(path_weight, path_val, num_test, all_or_every):
    root = path_val
    confidence = 0.5
    # imgs = list(sorted(os.listdir(os.path.join(root, image))))
    imgs = list(sorted(os.listdir(root)))
    print("ДЛИНА СПИСКА", len(imgs))
    count_masks_in_all_photo = []
    # all_or_every = 'all'

    if all_or_every == 'every':
        for i in range(0, len(imgs)):
            img_path = os.path.join(root, imgs[i])
            print("IMG_PATH", img_path)
            filename = f'/result_every{i}'
            print("FILE_NAME", filename)
            count_masks_in_one_photo,  x, y, x_1, y_1, masks = segment_instance(img_path, path_weight, num_test, filename, all_or_every, confidence=confidence)
            count_masks_in_all_photo.append(count_masks_in_one_photo)

        # print(f'EVERY {count_masks_in_all_photo[0]}')
        return count_masks_in_all_photo[0],  x, y, x_1, y_1, masks

    elif all_or_every == 'all':
        max_confidence = 0.0
        for i in range(0, len(imgs)):
            img_path = os.path.join(root, imgs[i])
            print("IMG_PATH", img_path)
            filename = f'/result_all{i}'
            print("FILE_NAME", filename)

            current_confidence, x, y, x_1, y_1, masks = segment_instance(img_path, path_weight, num_test, filename, all_or_every)

            # if current_confidence > max_confidence:
            #     max_confidence = current_confidence
            #     search_img_path = img_path
            #     search_filename = filename

        segment_instance(img_path, path_weight, num_test, filename, all_or_every, key=1)
        return 1, x, y, x_1, y_1, masks


# all_or_every = 'all'
# project_dir = Path(__file__).parent
# path_weight = project_dir / 'weights' /'model_12_43248_.pt/'
# path_val = project_dir / 'seg'
# path_res = project_dir / 'result'
# # path_res = 'C:/Users/EVM/Qt/pythonProject3/result/'
# root = project_dir / 'seg'
# number_of_segment,  x, y, x_1, y_1, masks = CNN(path_weight, path_val, path_res, all_or_every)
# print("Координаты", number_of_segment, x, y, x_1, y_1)
# all_or_every = 'every'
# path_weight_seg = project_dir / 'weights' / 'model_12_batch_4_.pt'
# number_of_segment,  x, y, x_1, y_1, masks = CNN(path_weight_seg, path_val, path_res, all_or_every)
# print("Кол-во сегментов", number_of_segment)
################
# if __name__ == "__main__":

# script, path_weight, path_val, path_res = argv
# #


# print("Mask after", mask_after.shape)
# # Дополнительная часть, чтобы посмотреть предсказанную маску (можно закоментить, чтобы часто не показывалось)
# rgb_mask = get_coloured_mask(mask_after)
#
# imgsq = sorted(os.listdir(path_val))
# imgs = os.path.join(path_val, imgsq[0])
#
# img = cv2.imread(imgs)[...,::-1]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.imread('C:/Users/EVM/Documents/camera_test/val/seg/normal_with_blick.jpg')[...,::-1]
# img = cv2.imread('C:/Users/EVM/Documents/camera_test/val/seg/detect_seg.jpg')[...,::-1]
# img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
# plt.imshow(img)
# plt.show()
