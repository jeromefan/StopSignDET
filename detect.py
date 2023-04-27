import cv2
import carla
import pygame
import argparse
from imutils.video import FPS
from CarlaEngine import CarlaClient


def get_args_parser():

    parser = argparse.ArgumentParser(
        '在Carla模拟器中生成一段视频以供后续使用',
        add_help=False
    )
    parser.add_argument(
        '-m', '--yolo-model',
        default='yolov8m',
        type=str,
        help='YOLO v8 模型 (n、s、m、l、x)'
    )
    parser.add_argument(
        '-l', '--video-length',
        default='500',
        type=int,
        help='视频长度 (帧数)'
    )
    parser.add_argument(
        '-fps', '--frames-per-second',
        default='60',
        type=int,
        help='FPS'
    )
    parser.add_argument(
        '-c', '--classes',
        nargs='+',
        type=int,
        default=[11],
        help='需要检测的类别 (默认值 11 是停车标志 Stop Sign)'
    )
    return parser


def main(args):
    try:
        carla_client = CarlaClient(
            yolo_model=args.yolo_model,
            frames_per_second=args.frames_per_second,
            classes=args.classes
        )
        pygame.init()

        carla_client.display = pygame.display.set_mode(
            (1280, 640),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame_clock = pygame.time.Clock()
        carla_client.setup_vehicle()
        carla_client.setup_camera()

        carla_client.set_synchronous_mode(True)
        carla_client.vehicle.set_autopilot(True)

        video_count = 0
        while video_count < args.video_length:
            spectator = carla_client.world.get_spectator()
            transform = carla_client.vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location + carla.Location(z=20),
                carla.Rotation(pitch=-90)
            ))

            fps = FPS().start()
            carla_client.world.tick()
            carla_client.capture = True
            pygame_clock.tick_busy_loop(args.frames_per_second)
            carla_client.render(carla_client.display)
            pygame.display.flip()
            pygame.event.pump()
            cv2.waitKey(1)
            fps.stop()
            video_count += 1

    except Exception as e:
        print(e)

    finally:
        try:
            carla_client.set_synchronous_mode(False)
            carla_client.vehicle.destroy()
            carla_client.camera.destroy()
            carla_client.output_video.release()
            carla_client.output_ori_video.release()
            carla_client.output_det_video.release()
            pygame.quit()
            cv2.destroyAllWindows()
        except:
            print('可能出错了！')
        print('退出模拟器！')


if __name__ == '__main__':

    args = get_args_parser()
    args = args.parse_args()
    main(args)
