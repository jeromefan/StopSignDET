import os
import cv2
import sys
import yaml
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from copy import deepcopy
from torchvision import transforms
from engine.misc import number_classes, class_names
sys.path.append('engine/yolov5')


def get_args_parser():

    parser = argparse.ArgumentParser('以Yolo v5为攻击模型, 生成有效的攻击',
                                     add_help=False)

    parser.add_argument('--epochs', default=1, type=int,
                        help='训练轮次')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--learning_rate', default=0.0001, type=float,
                        help='学习率')
    parser.add_argument('--image_size', default=640, type=int,
                        help='图像缩放大小')
    parser.add_argument('--hyp', default='engine/hyp.scratch.yaml', type=str,
                        help='hyperparameters path')

    return parser


def build_targets(detect_model, data, target_label):
    detections = detect_model(data)
    xywhn = detections.xywhn[0]
    targets = [0.0, target_label, xywhn[0, 0].item(
    ), xywhn[0, 1].item(), xywhn[0, 2].item(), xywhn[0, 3].item()]
    return torch.tensor(targets).unsqueeze(0)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    repo_loc = Path(os.path.dirname(
        os.path.abspath(__file__))+'/engine/yolov5/')
    detect_model = torch.hub.load(repo_or_dir=repo_loc,
                                  model='custom',
                                  path=Path('./weights/yolov5s.pt'),
                                  source='local',
                                  device=0 if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load('weights/yolov5s.pt', map_location='cpu')
    model = Model(ckpt['model'].yaml, ch=3, nc=80).to(device)

    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)
    nl = model.model[-1].nl
    hyp['box'] *= 3. / nl
    hyp['cls'] *= number_classes / 80. * 3. / nl
    hyp['obj'] *= (args.image_size / 640) ** 2 * 3. / nl
    model.nc = number_classes
    model.hyp = hyp
    model.gr = 1.0
    model.names = class_names

    loss_calculator = ComputeLoss(model)
    if not os.path.exists('results'):
        os.mkdir('results')

    patch_trans = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])

    trans = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    unloader = transforms.ToPILImage()

    # data
    patch_param_1 = patch_trans(Image.open(
        'assets/zidane.jpg')).unsqueeze(0).to(device)
    patch_param_2 = patch_trans(Image.open(
        'assets/zidane.jpg')).unsqueeze(0).to(device)
    data = trans(Image.open('assets/T_stop_d.TGA')).unsqueeze(0).to(device)

    targets = build_targets(detect_model, unloader(trans(data)), 40)
    targets = targets.to(device)

    for epoch in tqdm(range(50)):

        for i in range(450, 550):
            for j in range(300, 400):
                data[0, 0, j, i] = patch_param_1[0, 0, j-300, i-450]
                data[0, 1, j, i] = patch_param_1[0, 1, j-300, i-450]
                data[0, 2, j, i] = patch_param_1[0, 2, j-300, i-450]
        for i in range(150, 250):
            for j in range(300, 400):
                data[0, 0, j, i] = patch_param_2[0, 0, j-300, i-150]
                data[0, 1, j, i] = patch_param_2[0, 1, j-300, i-150]
                data[0, 2, j, i] = patch_param_2[0, 2, j-300, i-150]

        x_adv = data.detach().clone()
        x_adv.requires_grad = True
        y = model(x_adv)
        loss, _ = loss_calculator(y, targets)

        grad = torch.autograd.grad(
            outputs=loss, inputs=x_adv)[0].sign()

        patch_param_1 += 0.001 * grad[0, :, 300:400, 450:550]
        patch_param_2 += 0.001 * grad[0, :, 300:400, 150:250]

        if (epoch+1) in [2, 10, 20, 30, 40, 50]:
            img_new = unloader(
                deepcopy(data.detach().cpu().squeeze(0))

            )
            annotated_img = detect_model(img_new).render()[0]
            annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'results/an_{epoch+1}.png', annotated_img)

            img_new.save(f'results/test_{epoch+1}.png')


if __name__ == '__main__':
    from engine.yolov5.utils.loss import ComputeLoss
    from engine.yolov5.models.yolo import Model

    args = get_args_parser()
    args = args.parse_args()
    main(args)
