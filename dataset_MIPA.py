import json
import os
import random

import numpy as np
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


AFFORDANCE_LABELS = [
    "grasp",
    "contain",
    "lift",
    "open",
    "lay",
    "sit",
    "support",
    "wrapgrasp",
    "pour",
    "move",
    "display",
    "push",
    "listen",
    "wear",
    "press",
    "cut",
    "stab",
]


def pc_normalize(point_cloud):
    centroid = np.mean(point_cloud, axis=0)
    point_cloud = point_cloud - centroid
    scale = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
    return point_cloud / scale


class MIPA(Dataset):
    """MIPA loader for MIFAG and its multi-image ablations."""

    def __init__(
        self,
        run_type,
        setting,
        data_root=".",
        split_root="data/mipa_splits",
        image_count=2,
        image_size=(224, 224),
    ):
        super().__init__()
        if run_type not in {"train", "val"}:
            raise ValueError("run_type must be 'train' or 'val'")
        if image_count < 1:
            raise ValueError("image_count must be at least 1")

        self.run_type = run_type
        self.setting = setting.capitalize()
        self.data_root = data_root
        self.split_root = split_root
        self.image_count = image_count
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if run_type == "train":
            split_path = self._split_path(
                f"dataset_{self.setting.lower()}_train.json"
            )
            logger.info(f"Training split: {split_path}")
            split = self._read_json(split_path)
            self.samples = [
                (point_path, affordance)
                for point_path, affordance_to_images in split.items()
                for affordance in affordance_to_images
            ]
            self.image_pool = self._build_train_image_pool(split)
        else:
            split_path = self._split_path(
                f"dataset_{self.setting.lower()}_test_img_{image_count}.json"
            )
            logger.info(f"Evaluation split: {split_path}")
            split = self._read_json(split_path)
            self.samples = [(point_path, images) for point_path, images in split.items()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        point_path, image_spec = self.samples[index]
        split_name = "Train" if self.run_type == "train" else "Test"

        if self.run_type == "train":
            category = os.path.basename(os.path.dirname(point_path))
            available_images = self.image_pool[category][image_spec]
            image_names = self._sample_images(available_images, self.image_count)
        else:
            image_names = list(image_spec)
            if len(image_names) != self.image_count:
                raise ValueError(
                    f"Expected {self.image_count} images for {point_path}, "
                    f"found {len(image_names)} in the evaluation split."
                )

        images = []
        image_paths = []
        affordance = None
        for image_name in image_names:
            parts = image_name.split("_")
            category, affordance = parts[-3], parts[-2]
            relative_path = (
                f"Data/{self.setting}/Img/{split_name}/{category}/"
                f"{affordance}/{image_name}"
            )
            image_paths.append(relative_path)
            image = Image.open(self._data_path(relative_path)).convert("RGB")
            image = image.resize(self.image_size)
            images.append(self.image_transform(image))

        points, affordance_labels = self._read_point_file(point_path)
        points = pc_normalize(points).transpose()
        affordance_index = AFFORDANCE_LABELS.index(affordance)
        label = affordance_labels[:, affordance_index]

        if self.run_type == "train":
            return images, points, label, affordance_index
        return images, points, label, affordance_index, point_path, image_paths

    def _read_point_file(self, path):
        coordinates = []
        with open(self._data_path(path), "r") as point_file:
            for line in point_file:
                values = line.strip().split()
                coordinates.append([float(value) for value in values[2:]])
        data = np.asarray(coordinates)
        return data[:, :3], data[:, 3:]

    def _split_path(self, filename):
        return os.path.join(self.split_root, filename)

    def _data_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.data_root, path)

    @staticmethod
    def _read_json(path):
        with open(path, "r") as json_file:
            return json.load(json_file)

    @staticmethod
    def _sample_images(images, image_count):
        if len(images) < image_count:
            raise ValueError(
                f"Requested {image_count} images, but only {len(images)} are available."
            )
        return random.sample(images, image_count)

    @staticmethod
    def _build_train_image_pool(training_split):
        image_pool = {}
        for point_path, affordance_to_images in training_split.items():
            category = os.path.basename(os.path.dirname(point_path))
            category_pool = image_pool.setdefault(category, {})
            for affordance, image_names in affordance_to_images.items():
                bucket = category_pool.setdefault(affordance, [])
                for image_name in image_names:
                    if image_name not in bucket:
                        bucket.append(image_name)
        return image_pool
