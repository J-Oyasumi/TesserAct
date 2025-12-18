import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import json
import glob
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision.transforms as TT
from accelerate.logging import get_logger
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(__name__)

DATASET2ROBOT = {
    "fractal20220817_data": "google robot",
    "bridge": "Trossen WidowX 250 robot arm",
    "ssv2": "human hand",
    "rlbench": "Franka Emika Panda",
}
DATASET2RES = {
    "fractal20220817_data": (256, 320),
    # "fractal20220817_data_superres": (512, 640),
    "bridge": (480, 640),
    # "ssv2": (240, 320),
    "rlbench": (512, 512),
    # common resolutions
    # "480p": (480, 854),
    # "720p": (720, 1280),
}
HEIGHT_BUCKETS = [240, 256, 480, 720]
WIDTH_BUCKETS = [320, 426, 640, 854, 1280]
FRAME_BUCKETS = [9, 49, 100]


def crop_and_resize_frames(frames, target_size, interpolation="bilinear"):
    target_height, target_width = target_size
    original_height, original_width = frames[0].shape[:2]
    if original_height == target_height and original_width == target_width:
        return [frame for frame in frames]

    # ==== interpolation method ====
    if interpolation == "bilinear":
        interpolation = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation = cv2.INTER_NEAREST
    else:
        interpolation = cv2.INTER_LINEAR
        logger.warning(f"Unsupported interpolation: {interpolation}. Using bilinear instead.")

    processed_frames = []
    for frame in frames:
        original_height, original_width = frame.shape[:2]
        aspect_ratio_target = target_width / target_height
        aspect_ratio_original = original_width / original_height

        if aspect_ratio_original > aspect_ratio_target:
            new_width = int(aspect_ratio_target * original_height)
            start_x = (original_width - new_width) // 2
            cropped_frame = frame[:, start_x : start_x + new_width]
        else:
            new_height = int(original_width / aspect_ratio_target)
            start_y = (original_height - new_height) // 2
            cropped_frame = frame[start_y : start_y + new_height, :]
        resized_frame = cv2.resize(cropped_frame, (target_width, target_height), interpolation=interpolation)
        processed_frames.append(resized_frame)

    return processed_frames


class RoboDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.dataset_file = dataset_file
        self.caption_column = caption_column
        self.video_column = video_column
        self.max_num_frames = max_num_frames
        self.id_token = f"{id_token.strip()} " if id_token else ""
        self.height_buckets = height_buckets or HEIGHT_BUCKETS
        self.width_buckets = width_buckets or WIDTH_BUCKETS
        self.frame_buckets = frame_buckets or FRAME_BUCKETS
        self.load_tensors = load_tensors
        self.random_flip = random_flip
        self.image_to_video = image_to_video

        self.resolutions = [
            (f, h, w) for h in self.height_buckets for w in self.width_buckets for f in self.frame_buckets
        ]

        self._init_transforms()
        self._load_samples()

    def _init_transforms(self):
        """Initialize video transforms based on class requirements"""
        transform_list = [
            transforms.Lambda(self.identity_transform),
            transforms.Lambda(self.scale_transform),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]

        if self.random_flip:
            transform_list.insert(0, transforms.RandomHorizontalFlip(self.random_flip))

        self.video_transforms = transforms.Compose(transform_list)

    def _load_samples(self):
        """Load samples from dataset file or local paths"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            self.samples, test_samples = self._load_openx_dataset_from_local_path("bridge")
            logger.info(f"Loaded {len(self.samples)} train and {len(test_samples)} test samples from Bridge dataset.")

            # Save samples to dataset file
            random.shuffle(self.samples)
            with open(self.dataset_file, "w") as f:
                json.dump(self.samples, f)
            with open(self.dataset_file.replace(".json", "_test.json"), "w") as f:
                json.dump(test_samples, f)
        else:
            with open(self.dataset_file, "r") as f:
                self.samples = json.load(f)
            self._get_rlbench_instructions()

    @staticmethod
    def identity_transform(x):
        return x

    @staticmethod
    def scale_transform(x):
        return x / 255.0

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        for i in range(5):
            try:
                return self.getitem(index)
            except Exception as e:
                logger.error(f"Error loading sample {self.samples[index][1]}: {e}")
                index = random.randint(0, len(self.samples) - 1)
        return self.getitem(index)

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return index

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        image, video = self._preprocess_video(Path(sample[1]))
        instruction = self.get_instruction(index)

        return {
            "prompt": self.id_token + instruction,
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
        }

    def _train_test_split(self, samples: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
        if len(samples) > 4000:
            test_size = 200
        else:
            test_size = max(1, int(len(samples) * 0.05))

        indices = list(range(len(samples)))
        random.shuffle(indices)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        test_samples = [samples[i] for i in test_indices]
        train_samples = [samples[i] for i in train_indices]

        return train_samples, test_samples

    def _load_openx_dataset_from_local_path(self, dataname) -> Tuple[List[str], List[Path]]:
        samples = []
        for subdir in Path(f"data/{dataname}/processed").iterdir():
            if subdir.is_dir():
                rgb_dir = subdir.joinpath("video", "rgb.mp4")
                rgb_valid = rgb_dir.exists()
                if not rgb_dir.exists():
                    rgb_dir = subdir.joinpath("image", "rgb")
                    rgb_valid = rgb_dir.exists() and any(rgb_dir.glob("*.png"))
                if rgb_valid:
                    # Load prompt from instruction.txt if available
                    instruction_file = subdir.joinpath("instruction.txt")
                    if instruction_file.is_file():
                        instruction = instruction_file.read_text().strip()
                    else:
                        instruction = "null"
                    # path str
                    samples.append([instruction, str(rgb_dir)])

        train_samples, test_samples = self._train_test_split(samples)
        return train_samples, test_samples

    def _load_ssv2_dataset_from_local_path(self) -> Tuple[List[str], List[Path]]:
        labels_file = Path("data/ssv2/labels/train.json")
        video_root = Path("data/ssv2/20bn-something-something-v2")
        with labels_file.open("r", encoding="utf-8") as f:
            labels = json.load(f)
        samples = []
        for entry in labels:
            video_id = entry.get("id")
            label = entry.get("label", "null")
            video_path = video_root / f"{video_id}.webm"
            samples.append([label, str(video_path)])

        train_samples, test_samples = self._train_test_split(samples)
        return train_samples, test_samples

    def _get_rlbench_instructions(self) -> List[str]:
        self.rlbench_instructions = {}
        taskvar_json = Path("data/rlbench/taskvar_instructions.jsonl")
        if not taskvar_json.exists():
            logger.warning(f"Taskvar json {taskvar_json} does not exist.")
            return
        with jsonlines.open(taskvar_json, "r") as reader:
            for obj in reader:
                task = obj["task"]
                self.rlbench_instructions.setdefault(task, obj["variations"]["0"])

    def _load_rlbench_dataset_from_local_path(self) -> List[List[str]]:
        rlbench_path = Path("data/rlbench/train_dataset/microsteps/seed100")

        self._get_rlbench_instructions()

        samples = [
            [task_dir.name, str(rgb_path)]
            for task_dir in rlbench_path.iterdir()
            for episode_dir in task_dir.glob("variation0/episodes/*")
            for rgb_path in episode_dir.glob("video/*rgb.mp4")
        ]
        # find which path don't have video
        for task_dir in rlbench_path.iterdir():
            for episode_dir in task_dir.glob("variation0/episodes/*"):
                rgb_path = episode_dir.glob("video/*rgb.mp4")
                if not rgb_path:
                    print(f"Missing video: {episode_dir}")
        train_samples, test_samples = self._train_test_split(samples)
        return train_samples, test_samples

    def _adjust_num_frames(self, frames, target_num_frames=None):
        if target_num_frames is None:
            target_num_frames = self.max_num_frames
        frame_count = len(frames)
        if frame_count < target_num_frames:
            extra = target_num_frames - frame_count
            if isinstance(frames, list):
                frames.extend([frames[-1]] * extra)
            elif isinstance(frames, torch.Tensor):
                frame_to_add = [frames[-1]] * extra
                frames = [f for f in frames] + frame_to_add
        elif frame_count > target_num_frames:
            indices = np.linspace(0, frame_count - 1, target_num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        return frames

    def get_instruction(self, index: int) -> str:
        if random.random() < 0.05:
            instruction = ""
        else:
            sample = self.samples[index]
            instruction = sample[0].lower()
            path = sample[1]

            if "rlbench" in str(path):
                task_name = path.split("/")[5]
                instruction = random.choice(self.rlbench_instructions[task_name]) + f" {DATASET2ROBOT['rlbench']}"
            elif "fractal20220817_data" in str(path):
                instruction += f" {DATASET2ROBOT['fractal20220817_data']}"
            elif "bridge" in str(path):
                instruction += f" {DATASET2ROBOT['bridge']}"
            elif "ssv2" in str(path):
                instruction += f" {DATASET2ROBOT['ssv2']}"
            else:
                raise ValueError(f"Unknown dataset for path: {path}")

        return instruction

    def _read_rgb_data(self, path: Path) -> torch.Tensor:
        if path.is_dir():
            frames = self._read_video_from_dir(path, adjust_num_frames=False)
        elif path.suffix == ".webm" or path.suffix == ".mp4":
            frames = self._read_video_from_webm(path, adjust_num_frames=False)
        else:
            raise ValueError(f"Unsupported video format: {path}")
        return frames

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Loads a single video from either:
        - A directory of RGB frames, or
        - A single .webm video file.

        Returns:
            image: the first frame as an image if image_to_video=True, else None
            video: a tensor [F, C, H, W] of frames
            None for embeddings if load_tensors=False
        """
        if path.is_dir():
            frames = self._read_video_from_dir(path)
        elif path.suffix == ".webm" or path.suffix == ".mp4":
            frames = self._read_video_from_webm(path)
            if "ssv2" in str(path):
                frames = crop_and_resize_frames(frames, (256, 320))
        else:
            raise ValueError(f"Unsupported video format: {path}")
        # randome resize to other resolutions
        if random.random() < 0.2:
            target_size = random.choice(list(DATASET2RES.values()))
            frames = crop_and_resize_frames(frames, target_size)

        # transform frames to tensor
        frames = [self.video_transforms(torch.from_numpy(img).permute(2, 0, 1).float()) for img in frames]
        video = torch.stack(frames, dim=0)  # [F, C, H, W]
        image = video[:1].clone() if self.image_to_video else None

        return image, video

    def _read_video_from_dir(self, path: Path, adjust_num_frames: bool = True) -> List[np.ndarray]:
        assert path.is_dir(), f"Path {path} is not a directory."
        frame_paths = sorted(list(path.glob("*.png")), key=lambda x: int(x.stem))
        if adjust_num_frames:
            frame_paths = self._adjust_num_frames(frame_paths)
        frames = []
        for fp in frame_paths:
            img = np.array(Image.open(fp).convert("RGB"))
            frames.append(img)
        return frames

    def _read_video_from_webm(self, path: Path, adjust_num_frames: bool = True) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        if adjust_num_frames:
            frames = self._adjust_num_frames(frames)
        return frames


class RoboDepth(RoboDataset):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
        )

    def _load_samples(self):
        """Override to load additional datasets"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            bridge_train, bridge_test = self._load_openx_dataset_from_local_path("bridge")
            logger.info(f"Loaded {len(bridge_train)} train and {len(bridge_test)} test samples from Bridge dataset.")
            self.samples = bridge_train

            fractal_train, fractal_test = self._load_openx_dataset_from_local_path("fractal20220817_data")
            logger.info(
                f"Loaded {len(fractal_train)} train and {len(fractal_test)} test samples from Fractal20220817 dataset."
            )
            self.samples += fractal_train

            ssv2_train, ssv2_test = self._load_ssv2_dataset_from_local_path()
            logger.info(f"Loaded {len(ssv2_train)} train and {len(ssv2_test)} test samples from SSV2 dataset.")
            self.samples += ssv2_train

            # Combine all test samples
            test_samples = bridge_test + fractal_test + ssv2_test

            # Save samples to dataset files
            random.shuffle(self.samples)
            random.shuffle(test_samples)

            train_file = self.dataset_file
            test_file = str(self.dataset_file).replace(".json", "_test.json")

            with open(train_file, "w") as f:
                json.dump(self.samples, f)
            with open(test_file, "w") as f:
                json.dump(test_samples, f)
        else:
            with open(self.dataset_file, "r") as f:
                self.samples = json.load(f)

    def _read_depth_data(self, path: Path) -> torch.Tensor:
        """
        Reads a depth data file in .npz format and returns it as a [T, H, W] torch tensor.
        """
        assert path.is_file(), f"Depth file {path} does not exist."
        depth_array = np.load(path)["arr_0"].astype(np.float32)
        return depth_array

    def get_depth_data(self, rgb_dir, rgb_video, target_size) -> Tuple[torch.Tensor, bool]:
        depth_path = Path(str(rgb_dir).replace("video", "depth/npz").replace("rgb.mp4", "depth.npz"))

        if depth_path.exists():
            depth_video = self._read_depth_data(depth_path)  # [T, H, W]
            depth_video = (depth_video - depth_video.min()) / (depth_video.max() - depth_video.min() + 1e-8)
            if "rlbench" in str(rgb_dir):
                depth_video = 1 - depth_video
            depth_video *= 255.0
            depth_video = np.stack([depth_video] * 3, axis=-1)  # [T, H, W, 3]
            depth_video = crop_and_resize_frames(depth_video, target_size)
            depth_video = [
                self.video_transforms(torch.from_numpy(img).permute(2, 0, 1).float()).unsqueeze(0)
                for img in depth_video
            ]
            depth_video = torch.cat(depth_video, dim=0)  # [T, 3, H, W]
            if len(rgb_video) != len(depth_video):
                # logger.warning(f"RGB and depth video lengths do not match: {len(rgb_video)} vs {len(depth_video)}")
                logger.warning(f"{depth_path} RGB {len(rgb_video)} != DEPTH {len(depth_video)}")
                depth_video = self._adjust_num_frames(depth_video, len(rgb_video))
                depth_video = torch.stack(depth_video, dim=0)  # [T, 3, H, W]
            have_depth = True
        else:
            depth_video = torch.zeros_like(rgb_video)
            have_depth = False
        return depth_video, have_depth

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        """
        Overrides the parent class method to load both RGB and depth data and return a concatenated video.

        Returns:
            video: a tensor [T, H, W, 6] of concatenated RGB and depth frames.
        """
        target_size = random.choice(list(DATASET2RES.values()))

        # ==== Load RGB frames =====
        rgb_dir = path
        frames = self._read_rgb_data(rgb_dir)
        frames = crop_and_resize_frames(frames, target_size)
        rgb_video = [
            self.video_transforms(torch.from_numpy(img).permute(2, 0, 1).float()).unsqueeze(0) for img in frames
        ]
        rgb_video = torch.cat(rgb_video, dim=0)  # [T, 3, H, W]

        # ==== Load depth data ====
        depth_video, have_depth = self.get_depth_data(rgb_dir, rgb_video, target_size)
        concatenated_video = torch.cat((rgb_video, depth_video), dim=1)  # [T, 6, H, W]

        # Adjust frames to match max_num_frames
        concatenated_video = self._adjust_num_frames(list(concatenated_video))
        concatenated_video = torch.stack(concatenated_video, dim=0)  # [T, 6, H, W]
        image = concatenated_video[:1].clone()
        return image, concatenated_video, have_depth

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return index

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        image, video, have_depth = self._preprocess_video(Path(sample[1]))
        instruction = self.get_instruction(index)

        return {
            "prompt": self.id_token + instruction,
            "image": image,
            "video": video,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
            "path": sample[1],
            "have_depth": have_depth,
        }

    def load_one_sample(self, path: str):
        # for debug: robodataset.load_one_sample(path)
        image, video = self._preprocess_video(Path(path))
        return image, video


class RoboDepthNormal(RoboDepth):
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
        )

    def _load_samples(self):
        """Override to load additional datasets"""
        if self.dataset_file is None or not Path(self.dataset_file).exists():
            bridge_train, bridge_test = self._load_openx_dataset_from_local_path("bridge")
            logger.info(f"Loaded {len(bridge_train)} train and {len(bridge_test)} test samples from Bridge dataset.")
            self.samples = bridge_train
            test_samples = bridge_test

            # NOTE: uncomment this when you download and preprocess the dataset
            # fractal_train, fractal_test = self._load_openx_dataset_from_local_path("fractal20220817_data")
            # logger.info(f"Loaded {len(fractal_train)} train and {len(fractal_test)} test samples from Fractal20220817 dataset.")
            # self.samples += fractal_train
            # test_samples += fractal_test
            # rlbench_train, rlbench_test = self._load_rlbench_dataset_from_local_path()
            # logger.info(f"Loaded {len(rlbench_train)} train and {len(rlbench_test)} test samples from RLBench dataset.")
            # self.samples += rlbench_train
            # test_samples += rlbench_test

            # Save samples to dataset files
            random.shuffle(self.samples)

            train_file = self.dataset_file
            test_file = str(self.dataset_file).replace(".json", "_test.json")

            with open(train_file, "w") as f:
                json.dump(self.samples, f)
            with open(test_file, "w") as f:
                json.dump(test_samples, f)
        else:
            with open(self.dataset_file, "r") as f:
                self.samples = json.load(f)
            self._get_rlbench_instructions()

    def sample_target_size(self, path=None, raw_size=None):
        if raw_size is not None and random.random() < 0.5:
            return raw_size
        # inversely proportional to the 1.5th power of its area
        prob = {k: 1 / (v[0] * v[1]) ** 1.5 for k, v in DATASET2RES.items()}
        prob_sum = sum(prob.values())
        prob = {k: v / prob_sum for k, v in prob.items()}
        target_size = random.choices(list(prob.keys()), weights=list(prob.values()), k=1)[0]
        target_size = DATASET2RES[target_size]
        return target_size

    def get_normal_data(self, rgb_dir, rgb_video, target_size) -> Tuple[torch.Tensor, bool]:
        if "ssv2" in str(rgb_dir):
            normal_path = Path("not-exist")
        elif rgb_dir.is_dir():
            normal_path = Path(str(rgb_dir.parent).replace("rgb", "video/normal.mp4"))
        else:
            normal_path = Path(str(rgb_dir).replace("rgb.mp4", "normal.mp4"))
        if normal_path.exists():
            normal_frames = self._read_rgb_data(normal_path)
            normal_frames = crop_and_resize_frames(normal_frames, target_size)
            normal_video = [
                self.video_transforms(torch.from_numpy(img).permute(2, 0, 1).float()).unsqueeze(0)
                for img in normal_frames
            ]
            normal_video = torch.cat(normal_video, dim=0)  # [T, 3, H, W], range [-1, 1]
            if "rlbench" in str(rgb_dir):
                mask_video = torch.sum((normal_video + 1) ** 2, axis=1) < 0.02
                normal_video[:, 1] *= -1
                normal_video[:, 1][mask_video] = 0.0
            normal_mask = True
        else:
            normal_video = torch.zeros_like(rgb_video)
            normal_mask = False

        return normal_video, normal_mask

    def _preprocess_video(self, path: Path) -> torch.Tensor:
        """
        Overrides the parent class method to load both RGB and depth data and return a concatenated video.

        Returns:
            video: a tensor [T, H, W, 6] of concatenated RGB and depth frames.
        """
        # ==== Load RGB frames =====
        rgb_dir = path
        frames = self._read_rgb_data(rgb_dir)
        height, width = frames[0].shape[:2]
        target_size = self.sample_target_size(path, (height, width))
        target_size = [480, 640]
        # target_size = [256, 320]
        frames = crop_and_resize_frames(frames, target_size)
        rgb_video = [
            self.video_transforms(torch.from_numpy(img).permute(2, 0, 1).float()).unsqueeze(0) for img in frames
        ]
        rgb_video = torch.cat(rgb_video, dim=0)  # [T, 3, H, W]
        T, H, W = rgb_video.shape[0], rgb_video.shape[2], rgb_video.shape[3]
        rgb_mask = True

        # ==== Load depth data ====
        depth_video, depth_mask = self.get_depth_data(rgb_dir, rgb_video, target_size)

        # ==== Load normal data ====
        normal_video, normal_mask = self.get_normal_data(rgb_dir, rgb_video, target_size)

        # ==== Transform RGB and depth frames ====
        if 0 < abs(len(rgb_video) - len(normal_video)) < 2:
            normal_video = self._adjust_num_frames(normal_video, len(rgb_video))
            normal_video = torch.stack(normal_video, dim=0)
        concatenated_video = torch.cat((rgb_video, depth_video, normal_video), dim=1)  # [T, 9, H, W]

        # Adjust frames to match max_num_frames
        concatenated_video = self._adjust_num_frames(list(concatenated_video))
        concatenated_video = torch.stack(concatenated_video, dim=0)  # [T, 9, H, W]
        mask = [rgb_mask, depth_mask, normal_mask]
        image = concatenated_video[:1].clone()
        return image, concatenated_video, mask

    def getitem(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Special logic for bucket sampler
            return index

        sample = self.samples[index]
        image, video, mask = self._preprocess_video(Path(sample[1]))

        instruction = self.get_instruction(index)

        return {
            "prompt": self.id_token + instruction,
            "image": image,
            "video": video,
            "mask": mask,
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
            "path": sample[1],
        }


class RoboPointmap(RoboDataset):
    """
    Dataset for RGB + Pointmap data.
    Pointmap is stored as [T, H, W, 3] containing XYZ coordinates.
    """
    def __init__(
        self,
        data_root: str,
        dataset_file: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        max_num_frames: int = 49,
        id_token: Optional[str] = None,
        height_buckets: List[int] = None,
        width_buckets: List[int] = None,
        frame_buckets: List[int] = None,
        load_tensors: bool = False,
        random_flip: Optional[float] = None,
        image_to_video: bool = False,
    ) -> None:
        super().__init__(
            data_root=data_root,
            dataset_file=dataset_file,
            caption_column=caption_column,
            video_column=video_column,
            max_num_frames=max_num_frames,
            id_token=id_token,
            height_buckets=height_buckets,
            width_buckets=width_buckets,
            frame_buckets=frame_buckets,
            load_tensors=load_tensors,
            random_flip=random_flip,
            image_to_video=image_to_video,
        )

    def _read_pointmap_data(self, path: Path) -> np.ndarray:
        """
        Reads a pointmap data file in .npz format and returns it as a [T, H, W, 3] numpy array.

        Args:
            path: Path to the pointmap.npz file

        Returns:
            pointmap_array: [T, H, W, 3] array containing XYZ coordinates
        """
        assert path.is_file(), f"Pointmap file {path} does not exist."
        pointmap_array = np.load(path)["point_map"].astype(np.float32)
        mask = np.load(path)["mask"]
        # Check if All points are valid
        if not np.all(mask):
            logger.warning(f"Pointmap file {path} contains invalid points.")
            
        # Ensure shape is [T, H, W, 3]
        if len(pointmap_array.shape) == 3:
            # If shape is [T, H, W], expand to [T, H, W, 1] and repeat for 3 channels
            logger.warning(f"Pointmap has shape {pointmap_array.shape}, expected [T, H, W, 3]. Expanding to 3 channels.")
            pointmap_array = np.expand_dims(pointmap_array, axis=-1)
            pointmap_array = np.repeat(pointmap_array, 3, axis=-1)

        return pointmap_array

    def _load_and_normalize_pointmap(self, rgb_dir: Path, num_frames: int) -> Tuple[List[np.ndarray], bool]:
        """
        Load and normalize pointmap data while preserving spatial proportions.

        Args:
            rgb_dir: Path to RGB video file
            num_frames: Number of frames in RGB video

        Returns:
            pointmap_frames: List of normalized pointmap frames [H, W, 3] or None
            have_pointmap: Boolean indicating if pointmap exists
        """
        # Construct pointmap path: video/rgb.mp4 -> pointmap/npz/pointmap.npz
        pointmap_path = Path(str(rgb_dir).replace("video/rgb.mp4", "pointmap/npz/pointmap.npz"))

        # Also handle directory-based paths: image/rgb -> pointmap/npz/pointmap.npz
        if rgb_dir.is_dir():
            pointmap_path = Path(str(rgb_dir.parent.parent) + "/pointmap/npz/pointmap.npz")

        if not pointmap_path.exists():
            return None, False

        try:
            pointmap_array = self._read_pointmap_data(pointmap_path)  # [T, H, W, 3]

            # Normalize XYZ coordinates to [-1, 1] while preserving spatial proportions
            x_min, x_max = pointmap_array[..., 0].min(), pointmap_array[..., 0].max()
            y_min, y_max = pointmap_array[..., 1].min(), pointmap_array[..., 1].max()
            z_min, z_max = pointmap_array[..., 2].min(), pointmap_array[..., 2].max()

            # Calculate center and scale
            center = np.array([
                (x_max + x_min) / 2,
                (y_max + y_min) / 2,
                (z_max + z_min) / 2
            ])

            # Use the maximum range across all axes to preserve aspect ratio
            scale = max(x_max - x_min, y_max - y_min, z_max - z_min)

            # Normalize: center at origin, then scale to [-1, 1]
            if scale > 1e-8:
                pointmap_array = 2 * (pointmap_array - center) / scale
            else:
                pointmap_array = np.zeros_like(pointmap_array)

            # Adjust frame count to match RGB
            pointmap_frames = list(pointmap_array)  # Convert to list of [H, W, 3]
            if len(pointmap_frames) != num_frames:
                logger.warning(
                    f"{pointmap_path} RGB {num_frames} != POINTMAP {len(pointmap_frames)}"
                )
                pointmap_frames = self._adjust_num_frames(pointmap_frames, num_frames)

            return pointmap_frames, True

        except Exception as e:
            logger.error(f"Error loading pointmap from {pointmap_path}: {e}")
            return None, False

    def get_pointmap_data(self, rgb_dir: Path, rgb_video: torch.Tensor, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, bool]:
        """
        Load and process pointmap data corresponding to the RGB video.

        Args:
            rgb_dir: Path to RGB video file
            rgb_video: RGB video tensor [T, 3, H, W]
            target_size: Target (height, width) for resizing

        Returns:
            pointmap_video: Processed pointmap tensor [T, 3, H, W], normalized to [-1, 1]
            have_pointmap: Boolean indicating if pointmap data exists
        """
        # Construct pointmap path: video/rgb.mp4 -> pointmap/npz/pointmap.npz
        pointmap_path = Path(str(rgb_dir).replace("video/rgb.mp4", "pointmap/npz/pointmap.npz"))

        # Also handle directory-based paths: image/rgb -> pointmap/npz/pointmap.npz
        if rgb_dir.is_dir():
            pointmap_path = Path(str(rgb_dir.parent.parent) + "/pointmap/npz/pointmap.npz")

        if pointmap_path.exists():
            try:
                pointmap_array = self._read_pointmap_data(pointmap_path)  # [T, H, W, 3]

                # Normalize XYZ coordinates to [-1, 1] while preserving spatial proportions
                # Find global min/max for each axis across all frames
                x_min, x_max = pointmap_array[..., 0].min(), pointmap_array[..., 0].max()
                y_min, y_max = pointmap_array[..., 1].min(), pointmap_array[..., 1].max()
                z_min, z_max = pointmap_array[..., 2].min(), pointmap_array[..., 2].max()

                # Calculate center and scale
                center_x = (x_max + x_min) / 2
                center_y = (y_max + y_min) / 2
                center_z = (z_max + z_min) / 2
                center = np.array([center_x, center_y, center_z])

                # Use the maximum range across all axes to preserve aspect ratio
                scale = max(x_max - x_min, y_max - y_min, z_max - z_min)

                # Normalize: center at origin, then scale to [-1, 1]
                if scale > 1e-8:  # Avoid division by zero
                    pointmap_array = 2 * (pointmap_array - center) / scale
                else:
                    pointmap_array = np.zeros_like(pointmap_array)

                # Resize to target size
                pointmap_frames = crop_and_resize_frames(
                    [pointmap_array[t] for t in range(pointmap_array.shape[0])],
                    target_size,
                    interpolation="bilinear"
                )

                # Convert to tensor (already normalized to [-1, 1])
                # Note: We don't apply video_transforms because pointmap is already normalized
                pointmap_video = []
                for frame in pointmap_frames:
                    # Convert to torch and permute to [C, H, W]
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                    pointmap_video.append(frame_tensor.unsqueeze(0))

                pointmap_video = torch.cat(pointmap_video, dim=0)  # [T, 3, H, W]

                # Adjust frame count to match RGB video
                if len(rgb_video) != len(pointmap_video):
                    logger.warning(
                        f"{pointmap_path} RGB {len(rgb_video)} != POINTMAP {len(pointmap_video)}"
                    )
                    pointmap_video = self._adjust_num_frames(list(pointmap_video), len(rgb_video))
                    pointmap_video = torch.stack(pointmap_video, dim=0)  # [T, 3, H, W]

                have_pointmap = True

            except Exception as e:
                logger.error(f"Error loading pointmap from {pointmap_path}: {e}")
                pointmap_video = torch.zeros_like(rgb_video)
                have_pointmap = False
        else:
            # If pointmap doesn't exist, return zeros
            pointmap_video = torch.zeros_like(rgb_video)
            have_pointmap = False

        return pointmap_video, have_pointmap

    def _preprocess_video(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Overrides the parent class method to load both RGB and pointmap data.
        Ensures RGB and Pointmap undergo identical transformations (crop, resize, flip).

        Args:
            path: Path to RGB video file or directory

        Returns:
            image: First frame as image [1, 6, H, W]
            video: Concatenated RGB+Pointmap video [T, 6, H, W]
            have_pointmap: Boolean mask indicating if pointmap exists
        """
        # Sample target size (same for both RGB and pointmap)
        target_size = random.choice(list(DATASET2RES.values()))

        # Decide if we should flip (before any processing)
        should_flip = self.random_flip and random.random() < self.random_flip

        # ==== Load RGB frames =====
        rgb_dir = path
        rgb_frames = self._read_rgb_data(rgb_dir)  # [T, H, W, 3]

        # ==== Load pointmap data =====
        pointmap_frames, have_pointmap = self._load_and_normalize_pointmap(rgb_dir, len(rgb_frames))
        # pointmap_frames: [T, H, W, 3] or None

        # ==== Apply SAME crop and resize to both =====
        rgb_frames = crop_and_resize_frames(rgb_frames, target_size)
        if have_pointmap:
            pointmap_frames = crop_and_resize_frames(pointmap_frames, target_size)

        # ==== Apply SAME horizontal flip to both =====
        if should_flip:
            rgb_frames = [np.fliplr(frame) for frame in rgb_frames]
            if have_pointmap:
                pointmap_frames = [np.fliplr(frame) for frame in pointmap_frames]
                # CRITICAL: Flip X coordinate when flipping horizontally
                for frame in pointmap_frames:
                    frame[..., 0] *= -1

        # ==== Convert to tensors with appropriate transforms =====
        # RGB: apply standard video_transforms (scale + normalize)
        rgb_video = []
        for frame in rgb_frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            # Apply only scale and normalize (flip already done)
            frame_tensor = frame_tensor / 255.0  # [0, 1]
            frame_tensor = (frame_tensor - 0.5) / 0.5  # [-1, 1]
            rgb_video.append(frame_tensor.unsqueeze(0))
        rgb_video = torch.cat(rgb_video, dim=0)  # [T, 3, H, W]

        # Pointmap: convert to tensor (already normalized to [-1, 1])
        if have_pointmap:
            pointmap_video = []
            for frame in pointmap_frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
                pointmap_video.append(frame_tensor.unsqueeze(0))
            pointmap_video = torch.cat(pointmap_video, dim=0)  # [T, 3, H, W]
        else:
            pointmap_video = torch.zeros_like(rgb_video)

        # ==== Concatenate RGB and pointmap ====
        concatenated_video = torch.cat((rgb_video, pointmap_video), dim=1)  # [T, 6, H, W]

        # Adjust frames to match max_num_frames
        concatenated_video = self._adjust_num_frames(list(concatenated_video))
        concatenated_video = torch.stack(concatenated_video, dim=0)  # [T, 6, H, W]

        image = concatenated_video[:1].clone()  # [1, 6, H, W]

        return image, concatenated_video, have_pointmap

    def getitem(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample with RGB + Pointmap data.

        Returns:
            Dictionary containing:
                - prompt: Text instruction with robot name
                - image: First frame [1, 6, H, W]
                - video: Full video [T, 6, H, W]
                - mask: Boolean indicating if pointmap exists
                - video_metadata: Frame count and resolution
                - path: Path to the RGB video
        """
        if isinstance(index, list):
            # Special logic for bucket sampler
            return index

        if self.load_tensors:
            raise NotImplementedError("Loading tensors is not supported.")

        sample = self.samples[index]
        image, video, have_pointmap = self._preprocess_video(Path(sample[1]))
        instruction = self.get_instruction(index)

        return {
            "prompt": self.id_token + instruction,
            "image": image,
            "video": video,
            "mask": have_pointmap,  # Single boolean for pointmap
            "video_metadata": {
                "num_frames": video.shape[0],
                "height": video.shape[2],
                "width": video.shape[3],
            },
            "path": sample[1],
        }


class BucketSampler(Sampler):
    def __init__(
        self,
        data_source: RoboDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.buckets = {resolution: [] for resolution in data_source.resolutions}
        self._raised_warning_for_drop_last = False

    def __len__(self):
        if self.drop_last and not self._raised_warning_for_drop_last:
            self._raised_warning_for_drop_last = True
            logger.warning("Calculating the length for bucket sampler is not reliable when `drop_last=True`.")
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for index, data in enumerate(self.data_source):
            video_metadata = data["video_metadata"]
            f, h, w = (
                video_metadata["num_frames"],
                video_metadata["height"],
                video_metadata["width"],
            )

            self.buckets[(f, h, w)].append(data)
            if len(self.buckets[(f, h, w)]) == self.batch_size:
                if self.shuffle:
                    random.shuffle(self.buckets[(f, h, w)])
                yield self.buckets[(f, h, w)]
                del self.buckets[(f, h, w)]
                self.buckets[(f, h, w)] = []

        if self.drop_last:
            return

        for fhw, bucket in list(self.buckets.items()):
            if len(bucket) == 0:
                continue
            if self.shuffle:
                random.shuffle(bucket)
            yield bucket
            del self.buckets[fhw]
            self.buckets[fhw] = []


if __name__ == "__main__":
    import accelerate

    accelerator = accelerate.Accelerator()

    robodataset = RoboDepthNormal("data", dataset_file="cache/samples_depth_normal.json", max_num_frames=100)

    bucket_sampler = BucketSampler(robodataset, batch_size=4, shuffle=True, drop_last=False)
    dataloader = torch.utils.data.DataLoader(robodataset, batch_size=None, sampler=bucket_sampler, num_workers=96)
    dataloader = accelerator.prepare(dataloader)

    bar = tqdm(dataloader)
    for batch in bar:
        pass
