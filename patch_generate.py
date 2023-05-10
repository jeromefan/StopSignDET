import os
import sys
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.autograd import Variable
from engine.misc import square_transform


def get_args_parser():

    parser = argparse.ArgumentParser(
        '以Yolo v5为攻击模型, 生成 Adversarial Patch',
        add_help=False
    )
    parser.add_argument(
        '-s', '--img-size',
        default=640,
        type=int,
        help='图像缩放大小'
    )
    parser.add_argument(
        '-m', '--yolo-model',
        default='yolov5m',
        type=str,
        help='YOLO v5 模型 (n、s、m、l、x)'
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
        help='攻击目标类别'
    )

    return parser


def build_targets(model, data, target_label):
    model.eval()
    pred = non_max_suppression(model(data))
    xyxy = pred[0][0, :4]
    print(xyxy)
    xywhn = xyxy2xywhn(xyxy.unsqueeze(0)).squeeze(0)
    targets = [0.0, target_label, xywhn[0].item(), xywhn[1].item(),
               xywhn[2].item(), xywhn[3].item()]
    return torch.tensor(targets).unsqueeze(0)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('results'):
        os.mkdir('results')

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

    trans = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])
    unloader = transforms.ToPILImage()

    # data
    data = trans(Image.open('assets/T_stop_d.TGA')).unsqueeze(0).to(device)
    targets = build_targets(model, data, args.cls)
    patch_transformed, mask = square_transform(
        data.shape, args.img_size)
    targets, patch_transformed, mask = targets.to(device), patch_transformed.to(
        device), mask.to(device)
    data_with_patch = torch.mul(
        (1 - mask), data) + torch.mul(mask, patch_transformed)
    print(f'Target is: {targets}')

    model.eval()
    target_pred = (non_max_suppression(
        model(data_with_patch))[0].cpu()[0, 5]).item()

    save_name = None
    pbar = tqdm(range(args.max_iter))
    for count in pbar:

        if target_pred == args.cls:
            save_name = f'results/{count}.png'
            unloader(data_with_patch.squeeze(0)).save(save_name)
            break

        # Train
        model.train()
        data_with_patch = Variable(
            data_with_patch.data, requires_grad=True)
        data_with_patch = data_with_patch.to(device)
        target_loss, _ = compute_loss(model(data_with_patch), targets)
        target_loss.backward()

        patch_grad = data_with_patch.grad.data.clone()
        data_with_patch.grad.data.zero_()
        patch_transformed += 0.1 * patch_grad
        patch_transformed = torch.clamp(patch_transformed, min=0., max=1.)

        # Test
        model.eval()
        data_with_patch = torch.mul(
            (1-mask), data) + torch.mul(mask, patch_transformed)
        data_with_patch = data_with_patch.to(device)
        target_pred = non_max_suppression(model(data_with_patch))[0].cpu()
        if target_pred.shape != torch.Size([0, 6]):
            target_pred = (target_pred[0, 5]).item()
        else:
            target_pred = None

        pbar.set_description(
            f'pred: {target_pred}, loss: {target_loss.item()}')

    del model
    return save_name


def test(args, save_name):
    if save_name is not None:

        # model
        '''
            models.common.AutoShape
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
                               _verbose=False,
                               device='cpu')
        img = Image.open(save_name)
        annotated_img = Image.fromarray(
            model(img).render()[0].astype('uint8')).convert('RGB')
        annotated_img.save('results/det_plot.png')
        trans = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor()
        ])
        print(non_max_suppression(model(trans(img).unsqueeze(0))))
    else:
        print('Attack Failure!')


if __name__ == '__main__':
    sys.path.append('engine/yolov5')
    from engine.yolov5.utils.loss import ComputeLoss
    from engine.yolov5.utils.general import non_max_suppression, xyxy2xywhn

    torch.set_printoptions(precision=4, sci_mode=False)

    args = get_args_parser()
    args = args.parse_args()
    save_name = main(args)
    test(args, save_name)
