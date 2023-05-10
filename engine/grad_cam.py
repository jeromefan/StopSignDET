import sys
import time
import torch
import torchvision
import torch.nn.functional as F
sys.path.append('engine/yolov5')  # NOQA: E402
from engine.yolov5.utils.metrics import box_iou
from engine.yolov5.utils.general import xywh2xyxy


class GradCam:
    # 初始化，得到target_layer层
    def __init__(self, model, layer_name, cls=None):
        self.model = model
        self.model.requires_grad_(True)
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = find_yolo_layer(self.model, layer_name)
        # 获取forward过程中每层的输入和输出，用于对比hook是不是正确记录
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        self.cls = cls

    def forward(self, input_img):
        saliency_maps = []
        b, c, h, w = input_img.size()
        preds, logits, _ = self.model(input_img)
        _, logits = gradcam_nms(preds, logits)

        if self.cls is not None:
            score = None
            for logit in logits[0]:
                cur_score = logit[self.cls]
                if score is None:
                    score = cur_score
                if cur_score > score:
                    score = cur_score
            self.model.zero_grad()
            score.backward(retain_graph=True)
            gradients = self.gradients['value']
            activations = self.activations['value']
            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations).sum(1, keepdim=True)
            saliency_map = F.relu(saliency_map)
            saliency_map = F.interpolate(saliency_map, size=(
                h, w), mode='bilinear', align_corners=False)
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (
                saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
            return saliency_map
        else:
            for logit in logits[0]:
                score = logit.max()
                self.model.zero_grad()
                score.backward(retain_graph=True)
                gradients = self.gradients['value']
                activations = self.activations['value']
                b, k, u, v = gradients.size()
                alpha = gradients.view(b, k, -1).mean(2)
                weights = alpha.view(b, k, 1, 1)
                saliency_map = (weights * activations).sum(1, keepdim=True)
                saliency_map = F.relu(saliency_map)
                saliency_map = F.interpolate(saliency_map, size=(
                    h, w), mode='bilinear', align_corners=False)
                saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
                saliency_map = (
                    saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
                saliency_maps.append(saliency_map)
            return saliency_maps

    def __call__(self, input_img):
        return self.forward(input_img)


def find_yolo_layer(model, layer_name):
    """Find yolov5 layer to calculate GradCAM and GradCAM++

    Args:
        model: yolov5 model.
        layer_name (str): the name of layer with its hierarchical information.

    Return:
        target_layer: found layer
    """
    hierarchy = layer_name.split('_')
    target_layer = model._modules[hierarchy[0]]

    for h in hierarchy[1:]:
        target_layer = target_layer._modules[h]
    return target_layer


def gradcam_nms(prediction, logits, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False,
                multi_label=False, labels=(), max_det=300):
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)
              ] * prediction.shape[0]
    logits_output = [torch.zeros(
        (0, nc), device=logits.device)] * logits.shape[0]
    # logits_output = [torch.zeros((0, 80), device=logits.device)] * logits.shape[0]
    for xi, (x, log_) in enumerate(zip(prediction, logits)):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        log_ = log_[xc[xi]]
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat(
                (box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]
            log_ = log_[conf.view(-1) > conf_thres]
        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        logits_output[xi] = log_[i]
        assert log_[i].shape[0] == x[i].shape[0]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output, logits_output
