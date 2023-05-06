import os
import cv2
import sys
import time
import carla
import torch
import random
import pygame
import weakref
import ultralytics
import numpy as np
from pathlib import Path

sys.path.append('engine/yolov5')


class CarlaClient():
    def __init__(
        self,
        yolo_model_choice,
        frames_per_second,
        classes
    ):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.capture = True
        self.image = None
        self.classes = classes

        self.yolo_model_choice = yolo_model_choice
        if 'yolov8' in yolo_model_choice:
            ultralytics.checks()
            self.yolo_model = ultralytics.YOLO(
                f'weights/{yolo_model_choice}.pt')
        else:
            repo_loc = Path(os.path.dirname(
                os.path.abspath(__file__))+'/yolov5/')
            self.yolo_model = torch.hub.load(repo_or_dir=repo_loc,
                                             model='custom',
                                             path=Path(
                                                 f'./weights/{yolo_model_choice}.pt'),
                                             source='local',
                                             device=0 if torch.cuda.is_available() else 'cpu')
            self.yolo_model.classes = classes

        if not os.path.exists('results'):
            os.mkdir('results')
        VideoName = time.strftime('%y%m%d-%H%M', time.localtime())
        self.output_video = cv2.VideoWriter(
            f'results/{VideoName}.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            frames_per_second, (1280, 640)
        )
        self.output_ori_video = cv2.VideoWriter(
            f'results/{VideoName}_ori.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            frames_per_second, (640, 640)
        )
        self.output_det_video = cv2.VideoWriter(
            f'results/{VideoName}_det.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            frames_per_second, (640, 640)
        )

    def setup_vehicle(self):
        vehicle = self.blueprint_library.find('vehicle.audi.etron')
        self.vehicle = self.world.spawn_actor(
            vehicle,
            self.map.get_spawn_points()[129]
        )

    def setup_camera(self):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '640')

        self.camera = self.world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=1.5, z=2.4)),
            attach_to=self.vehicle
        )
        weak_self = weakref.ref(self)
        self.camera.listen(
            lambda image: weak_self().set_image(weak_self, image)
        )

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    @staticmethod
    def set_image(weak_self, img):
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

    def render(self, display):
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            ori_frame = array[:, :, :3]

            if 'yolov8' in self.yolo_model_choice:
                yolo_result = self.yolo_model.predict(
                    source=ori_frame,
                    classes=self.classes,
                    verbose=False
                )
                annotated_frame = yolo_result[0].plot()
            else:
                annotated_frame = self.yolo_model(ori_frame).render()[0]

            img = np.hstack((ori_frame, annotated_frame))
            self.output_video.write(img)
            self.output_ori_video.write(ori_frame)
            self.output_det_video.write(annotated_frame)

            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(img_display.swapaxes(0, 1))
            display.blit(surface, (0, 0))
