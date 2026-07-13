import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from dataset_MIPA import MIPA
from model.MIFAG import get_MIFAG
from utils.eval import compute_metrics


def read_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def read_point_coordinates(path):
    coordinates = []
    with open(path, "r") as point_file:
        for line in point_file:
            values = line.strip().split()
            coordinates.append([float(value) for value in values[2:5]])
    return np.asarray(coordinates)


def save_point_cloud(prediction, point_path, reference_image, output_dir, data_root):
    import open3d as o3d

    resolved_point_path = (
        point_path if os.path.isabs(point_path) else os.path.join(data_root, point_path)
    )
    coordinates = read_point_coordinates(resolved_point_path)
    scores = prediction.squeeze().cpu().numpy()
    background = np.array([190, 190, 190])
    foreground = np.array([255, 0, 0])
    colors = (foreground - background) * scores[:, None] + background

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coordinates)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)

    object_name = os.path.basename(point_path).split("_")[-2]
    affordance = reference_image.split("_")[-2]
    sample_id = os.path.splitext(os.path.basename(point_path))[0].split("_")[-1]
    prediction_dir = os.path.join(output_dir, "pred")
    os.makedirs(prediction_dir, exist_ok=True)
    output_path = os.path.join(
        prediction_dir, f"{object_name}_{affordance}_{sample_id}_Pred.ply"
    )
    o3d.io.write_point_cloud(output_path, point_cloud)


def build_model(config, model_args):
    return get_MIFAG(
        pre_train=False,
        N_p=config.get("n_p", 64),
        emb_dim=config.get("emb_dim", 512),
        N_raw=config.get("n_raw", 2048),
        args=model_args,
    )


@torch.no_grad()
def evaluate(options):
    config = read_config(options.config)
    setting = config["setting"].capitalize()
    data_root = options.data_root or config.get("data_root", ".")
    split_root = options.split_root or config.get("split_root", "data/mipa_splits")
    image_count = config.get("train_img_nums", 2)
    batch_size = options.batch_size or config.get("batch_size", 64)

    model_args = SimpleNamespace(
        variant=config.get("variant", "full"),
        train_img_nums=image_count,
        invariant_extract_layers=config["invariant_extract_layers"],
        droprate=config.get("droprate", 0.0),
    )

    if options.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(options.device)

    dataset = MIPA(
        "val",
        setting,
        data_root=data_root,
        split_root=split_root,
        image_count=image_count,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=options.num_workers,
        shuffle=False,
    )

    model = build_model(config, model_args)
    checkpoint = torch.load(options.ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()

    output_dir = os.path.join(
        options.output_dir, os.path.splitext(os.path.basename(options.ckpt_path))[0]
    )
    os.makedirs(output_dir, exist_ok=True)

    predictions = []
    targets = []
    for images, points, labels, _, point_paths, image_paths in data_loader:
        images = [image.to(device) for image in images]
        points = points.float().to(device)
        output = model(images, points)
        prediction = output[0] if isinstance(output, tuple) else output
        predictions.append(prediction.cpu().numpy())
        targets.append(labels.float().unsqueeze(-1).numpy())

        if options.save_ply:
            for index, point_path in enumerate(point_paths):
                save_point_cloud(
                    prediction[index],
                    point_path,
                    image_paths[0][index],
                    output_dir,
                    data_root,
                )

    metrics = compute_metrics(np.concatenate(predictions), np.concatenate(targets))
    logger.info(
        "AUC:{AUC:.6f} | aIOU:{aIOU:.6f} | SIM:{SIM:.6f} | MAE:{MAE:.6f}",
        **metrics,
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--output_dir", default="eval_res")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_ply", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything(42)
    evaluate(parse_args())
