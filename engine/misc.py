import torch
import cv2
import numpy as np


def square_transform(data_shape, xyxy):
    patch = np.random.rand(1, 3, 120, 120)
    patch_transformed = np.zeros(data_shape)
    m_size = 120
    batch_size = patch_transformed.shape[0]
    for i in range(batch_size):
        # random rotation
        rot = np.random.choice(4)
        for j in range(3):
            patch[0][j] = np.rot90(patch[0][j], rot)
        # random location
        random_x = np.random.choice(
            [x for x in range(xyxy[0], xyxy[2] - m_size + 1)])
        if random_x + m_size > xyxy[2]:
            while random_x + m_size > xyxy[2]:
                random_x = np.random.choice(
                    [x for x in range(xyxy[0], xyxy[2] - m_size + 1)])
        random_y = np.random.choice(
            [y for y in range(xyxy[1], xyxy[3] - m_size + 1)])
        if random_y + m_size > xyxy[3]:
            while random_y + m_size > xyxy[3]:
                random_y = np.random.choice(
                    [y for y in range(xyxy[1], xyxy[3] - m_size + 1)])
        # apply patch
        patch_transformed[i][0][random_x:random_x+m_size,
                                random_y:random_y+m_size] = patch[0][0]
        patch_transformed[i][1][random_x:random_x+m_size,
                                random_y:random_y+m_size] = patch[0][1]
        patch_transformed[i][2][random_x:random_x+m_size,
                                random_y:random_y+m_size] = patch[0][2]
    mask = np.copy(patch_transformed)
    mask[mask != 0] = 1.0
    patch_transformed, mask = torch.FloatTensor(
        patch_transformed), torch.FloatTensor(mask)
    return patch_transformed, mask


# number of classes
number_classes = 80

# class names
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']
