import os
import yaml
import torch
import argparse
from PIL import Image
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms

from engine.misc import number_classes, class_names
import sys
sys.path.append('engine/yolov5')


def get_args_parser():

    parser = argparse.ArgumentParser('以Yolo v5为攻击模型, 生成有效的攻击',
                                     add_help=False)

    parser.add_argument('--epochs', default=1, type=int,
                        help='训练轮次')
    parser.add_argument('--batch-size', default=16, type=int,
                        help='batch size')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='学习率')
    parser.add_argument('--image_size', default=640, type=int,
                        help='图像缩放大小')
    parser.add_argument('--t', type=float, default=0.0001)
    parser.add_argument('--hyp', default='engine/hyp.scratch.yaml', type=str,
                        help='hyperparameters path')

    return parser


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    unloader = transforms.ToPILImage()

    # data
    patch_param = torch.autograd.Variable(torch.randn(
        3, 50, 50, device=device), requires_grad=True)
    optim = Adam([patch_param], lr=args.learning_rate)
    patch = unloader(patch_param.detach().cpu())
    data = Image.open('assets/T_stop_d.TGA')

    # model
    repo_loc = Path(os.path.dirname(
        os.path.abspath(__file__))+'/engine/yolov5/')
    model = torch.hub.load(repo_or_dir=repo_loc,
                           model='custom',
                           path=Path('./weights/yolov5x.pt'),
                           source='local',
                           device=0 if torch.cuda.is_available() else 'cpu')

    loss_calculator = ComputeLoss(model.model.model)

    for i in range(350, 400):
        for j in range(400, 450):
            data.putpixel((i, j), patch.getpixel((i-350, j-400)))
    out = model(data)
    print(out.pred)
    y = model(trans(data).unsqueeze(0))
    label_1 = torch.zeros((1, 6), device=device)
    label_1[0][1] = 11.0
    label_1[0][2] = out.pred[0][0][0]
    label_1[0][3] = out.pred[0][0][1]
    label_1[0][4] = out.pred[0][0][2]
    label_1[0][5] = out.pred[0][0][3]

    # out.pred[0]

    label_2 = non_max_suppression(y)[0]
    label_2_2 = non_max_suppression(y)[0]
    label_2_2[0][5] = 0

    print('labels:')
    print(label_1)
    print(label_2)
    print(label_2_2)

    loss_1 = loss_calculator(out.pred, label_1)
    loss_2 = loss_calculator(y, label_2)
    loss_2_2 = loss_calculator(y, label_2_2)

    print('loss:')
    # print(loss_1)
    print(loss_2)
    print(loss_2_2)

    # loss, _ = compute_loss(out, label_tensor.to(device))
    # optim.zero_grad()
    # loss.backward(retain_graph=True)
    # optim.step()

    # for i in range(350, 400):
    #     for j in range(400, 450):
    #         data[0][j][i] = patch[0][j-400][i-350]
    #         data[1][j][i] = patch[1][j-400][i-350]
    #         data[2][j][i] = patch[2][j-400][i-350]
    # unloader(data).save('test.png')


if __name__ == '__main__':
    from engine.yolov5.utils.loss import ComputeLoss
    from engine.yolov5.utils.general import non_max_suppression

    args = get_args_parser()
    args = args.parse_args()
    main(args)
