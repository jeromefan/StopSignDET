import torch
import cv2
import numpy as np


def square_transform(data_shape, img_size):
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
        random_x = np.random.choice(img_size)
        if random_x + m_size > img_size:
            while random_x + m_size > img_size:
                random_x = np.random.choice(img_size)
        random_y = np.random.choice(img_size)
        if random_y + m_size > img_size:
            while random_y + m_size > img_size:
                random_y = np.random.choice(img_size)
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
