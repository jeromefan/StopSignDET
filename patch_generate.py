import yaml
import torch
import argparse
from PIL import Image
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms
from engine.loss_compute import ComputeLoss
from engine.model_choice.yolov5 import utils
from engine.misc import number_classes, class_names


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
    device = torch.device('cuda')
    utils.notebook_init()

    # data
    patch = torch.randn(3, 50, 50, device=device)
    patch_param = torch.autograd.Variable(patch, requires_grad=True)
    optim = Adam([patch_param], lr=args.learning_rate)

    trans = transforms.ToTensor()
    data = trans(Image.open('assets/T_stop_d.TGA'))

    # model
    repo_loc = Path(
        '/home/ubuntu/workspace/StopSignDET/engine/model_choice/yolov5/')
    model = torch.hub.load(repo_or_dir=repo_loc,
                           model='custom',
                           path=Path('./weights/yolov5x.pt'),
                           source='local',
                           device=0)

    # # model parameters
    # with open(args.hyp) as f:
    #     hyp = yaml.safe_load(f)
    # nl = model.model.model.model[-1].nl
    # hyp['box'] *= 3. / nl  # scale to layers
    # hyp['cls'] *= number_classes / 80. * 3. / nl  # scale to classes and layers
    # # scale to image size and layers
    # hyp['obj'] *= (args.image_size / 640) ** 2 * 3. / nl
    # model.nc = number_classes  # attach number of classes to model
    # model.hyp = hyp  # attach hyperparameters to model
    # model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.names = class_names

    # # Start training
    # compute_loss = ComputeLoss(model)  # init loss class

    model.eval()

    for i in range(350, 400):
        for j in range(400, 450):
            data[0][j][i] = patch[0][j-400][i-350]
            data[1][j][i] = patch[1][j-400][i-350]
            data[2][j][i] = patch[2][j-400][i-350]

    # label_tensor = torch.zeros((1, 1, 1))  # 1个目标，5个元素
    # label_tensor[0, 0, 0] = 0  # 类别索引
    print(data.shape)
    print(data.unsqueeze(0).shape)
    out = model(data.unsqueeze(0))
    print(out.shape)
    out.render()

    # loss, _ = compute_loss(out, label_tensor.to(device))
    # optim.zero_grad()
    # loss.backward(retain_graph=True)
    # optim.step()

    # for i in range(350, 400):
    #     for j in range(400, 450):
    #         data[0][j][i] = patch[0][j-400][i-350]
    #         data[1][j][i] = patch[1][j-400][i-350]
    #         data[2][j][i] = patch[2][j-400][i-350]
    # unloader = transforms.ToPILImage()
    # unloader(data).save('test.png')


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)
