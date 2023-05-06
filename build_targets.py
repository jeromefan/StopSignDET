import os
import sys
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms
sys.path.append('engine/yolov5')


def main():
    trans = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    unloader = transforms.ToPILImage()

    # data
    data = Image.open('assets/T_stop_d.TGA')

    # model
    repo_loc = Path(os.path.dirname(
        os.path.abspath(__file__))+'/engine/yolov5/')
    model = torch.hub.load(repo_or_dir=repo_loc,
                           model='custom',
                           path=Path('./weights/yolov5s.pt'),
                           source='local',
                           device=0 if torch.cuda.is_available() else 'cpu')

    detections = model(unloader(trans(data)))
    print(detections.xywhn)
    print(detections.xywhn[0][0, 1])


if __name__ == '__main__':
    main()
