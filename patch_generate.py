import os
import sys
import time
import torch
import argparse

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from copy import deepcopy
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from engine.misc import build_targets, square_transform
sys.path.append('engine/yolov5')  # NOQA: E402
from engine.yolov5.utils.loss import ComputeLoss

torch.set_printoptions(precision=4, sci_mode=False)


def get_args_parser():

    parser = argparse.ArgumentParser(
        '以Yolo v5为攻击模型, 生成 Adversarial Patch',
        add_help=False
    )
    parser.add_argument(
        '-img', '--img-path',
        default='assets/T_stop_d.TGA',
        type=str,
        help='图像读取路径'
    )
    parser.add_argument(
        '-s', '--img-size',
        default=640,
        type=int,
        help='图像缩放大小'
    )
    parser.add_argument(
        '-m', '--yolo-model',
        default='yolov5s',
        type=str,
        help='YOLO v5 模型 (n、s、m、l、x)'
    )
    parser.add_argument(
        '-lr', '--learning-rate',
        default=0.1,
        type=float,
        help='迭代步长'
    )
    parser.add_argument(
        '-c', '--cls',
        default=1,
        type=int,
        help='攻击目标类别'
    )
    parser.add_argument(
        '-i', '--max-iter',
        default=1000,
        type=int,
        help='最大迭代次数'
    )

    return parser


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('results'):
        os.mkdir('results')
    tb_name = time.strftime('%y%m%d-%H%M', time.localtime())
    writer = SummaryWriter(f'results/{tb_name}')

    # model
    '''
        models.common.DetectMultiBackend
            models.yolo.DetectionModel
                torch.nn.modules.container.Sequential
    '''
    repo_loc = Path(os.path.dirname(
        os.path.abspath(__file__))+'/engine/yolov5/')
    model = torch.hub.load(repo_or_dir=repo_loc,
                           model='custom',
                           path=Path(f'./weights/{args.yolo_model}.pt'),
                           source='local',
                           autoshape=False,
                           device=0 if torch.cuda.is_available() else 'cpu')
    compute_loss = ComputeLoss(model.model)

    '''
        models.common.AutoShape
            models.common.DetectMultiBackend
                models.yolo.DetectionModel
                    torch.nn.modules.container.Sequential
    '''
    detect_model = torch.hub.load(repo_or_dir=repo_loc,
                                  model='custom',
                                  path=Path(f'./weights/{args.yolo_model}.pt'),
                                  source='local',
                                  _verbose=False,
                                  device='cpu')
    detect_model.max_det = 1

    loader = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    unloader = transforms.ToPILImage()

    # data
    data = loader(Image.open(args.img_path)).unsqueeze(0)
    targets = build_targets(detect_model, data, args.cls)
    patch_transformed, mask = square_transform(
        deepcopy(detect_model.model.model), data, data.shape)
    data, targets, patch_transformed, mask = data.to(device), targets.to(
        device), patch_transformed.to(device), mask.to(device)
    print(f'攻击目标为: {targets}')

    lr = args.learning_rate
    pbar = tqdm(range(args.max_iter))
    for i in pbar:

        # Test
        detect_model.eval()
        data_with_patch = torch.mul(
            (1-mask), data) + torch.mul(mask, patch_transformed)
        detection = detect_model(unloader(data_with_patch.squeeze(0)))
        data_with_patch = data_with_patch.to(device)
        target_pred = detection.pred[0][0, 5].item()
        target_conf = detection.pred[0][0, 4].item()
        writer.add_scalars(
            'test', {'pred': target_pred, 'conf': target_conf}, i + 1)

        if target_pred == args.cls:
            break

        # Train
        model.train()
        data_with_patch = Variable(
            data_with_patch.data, requires_grad=True)
        data_with_patch = data_with_patch.to(device)
        target_loss, _ = compute_loss(model(data_with_patch), targets)
        target_loss.backward()
        patch_transformed -= lr * data_with_patch.grad.data.sign()
        if i % 100 == 0:
            lr = lr * 0.9
        data_with_patch.grad.data.zero_()
        writer.add_scalar('train', target_loss.item(), i + 1)

        pbar.set_description(
            f'pred cls is {target_pred} with conf {target_conf}, loss: {target_loss.item()}')

    data_with_patch = torch.mul(
        (1-mask), data) + torch.mul(mask, patch_transformed)
    detection = detect_model(unloader(data_with_patch.squeeze(0)))
    target_pred = detection.pred[0][0, 5].item()
    print(f'最终攻击结果: {target_pred}')
    annotated_img = Image.fromarray(
        detection.render()[0].astype('uint8')).convert('RGB')
    annotated_img.save(f'results/{tb_name}.png')

    writer.flush()
    writer.close()


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)
