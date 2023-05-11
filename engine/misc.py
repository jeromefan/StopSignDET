import sys
import torch
import numpy as np
sys.path.append('engine/yolov5')  # NOQA: E402
from engine.yolov5.utils.general import non_max_suppression, xyxy2xywhn


def build_targets(model, data, target_label):
    pred = non_max_suppression(model(data), max_det=1)
    xyxy = pred[0][0, :4]
    print(f'未归一化的xyxy框: {xyxy}')
    xywhn = xyxy2xywhn(xyxy.unsqueeze(0)).squeeze(0)
    targets = [0.0, target_label, xywhn[0].item(), xywhn[1].item(),
               xywhn[2].item(), xywhn[3].item()]
    return torch.tensor(targets).unsqueeze(0)


def square_transform(heatmap, data_shape, patch_size):

    x = torch.argmax(heatmap).item() // data_shape[-1]
    y = torch.argmax(heatmap).item() % data_shape[-1]

    patch = np.random.rand(1, 3, patch_size, patch_size)
    patch_transformed = np.zeros(data_shape)
    m_size = patch_size
    batch_size = patch_transformed.shape[0]
    for i in range(batch_size):
        # random rotation
        rot = np.random.choice(4)
        for j in range(3):
            patch[0][j] = np.rot90(patch[0][j], rot)
        # random location
        while x - m_size // 2 < 0:
            x += 1
        while x + m_size // 2 > data_shape[-1]:
            x -= 1
        while y - m_size // 2 < 0:
            y += 1
        while y + m_size // 2 > data_shape[-1]:
            y -= 1
        # apply patch
        patch_transformed[i][0][x - m_size // 2:x + m_size // 2,
                                y - m_size // 2:y + m_size // 2] = patch[0][0]
        patch_transformed[i][1][x - m_size // 2:x + m_size // 2,
                                y - m_size // 2:y + m_size // 2] = patch[0][1]
        patch_transformed[i][2][x - m_size // 2:x + m_size // 2,
                                y - m_size // 2:y + m_size // 2] = patch[0][2]
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
