import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_MIPA import MIPA
from model.MIFAG import get_MIFAG
from utils.eval import compute_metrics
from utils.loss import HM_Loss


def read_config(path):
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_options(options, config):
    defaults = {
        "data_root": ".",
        "split_root": "data/mipa_splits",
        "resnet18_path": "pretrained/resnet18.pth",
        "variant": "full",
        "train_img_nums": 2,
        "invariant_extract_layers": 4,
        "droprate": 0.0,
        "batch_size": 64,
        "lr": 4e-5,
        "epochs": 80,
        "weight_decay": 1e-3,
        "loss_cls": 0.3,
        "loss_sim": 0.05,
        "num_workers": 8,
    }
    for name, default in defaults.items():
        cli_value = getattr(options, name)
        setattr(options, name, config.get(name, default) if cli_value is None else cli_value)
    options.setting = config["setting"].capitalize()
    return options


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model(options, config, pre_train=True):
    return get_MIFAG(
        img_model_path=options.resnet18_path,
        pre_train=pre_train,
        N_p=config.get("n_p", 64),
        emb_dim=config.get("emb_dim", 512),
        N_raw=config.get("n_raw", 2048),
        args=options,
    )


def model_loss(model, batch, device, heatmap_loss, classification_loss, options):
    images, points, labels, affordance_labels = batch
    images = [image.to(device) for image in images]
    points = points.float().to(device)
    labels = labels.float().unsqueeze(-1).to(device)
    affordance_labels = affordance_labels.to(device)

    output = model(images, points)
    if options.variant in {"full", "iam_only"}:
        prediction, similarity_loss, logits = output
        loss_hm = heatmap_loss(prediction, labels)
        loss_cls = classification_loss(logits, affordance_labels)
        loss = loss_hm + options.loss_cls * loss_cls
        components = {"heatmap": loss_hm, "classification": loss_cls}
        if options.train_img_nums > 1:
            similarity_loss = torch.mean(similarity_loss)
            loss = loss + options.loss_sim * similarity_loss
            components["similarity"] = similarity_loss
    else:
        prediction = output
        loss_hm = heatmap_loss(prediction, labels)
        loss = loss_hm
        components = {"heatmap": loss_hm}
    return loss, components


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    targets = []
    for images, points, labels, _, _, _ in data_loader:
        images = [image.to(device) for image in images]
        points = points.float().to(device)
        output = model(images, points)
        prediction = output[0] if isinstance(output, tuple) else output
        predictions.append(prediction.cpu().numpy())
        targets.append(labels.float().unsqueeze(-1).numpy())
    return compute_metrics(np.concatenate(predictions), np.concatenate(targets))


def restore_latest(run_dir, model, optimizer, scheduler, device):
    latest_files = glob.glob(os.path.join(run_dir, "latest_e*.pt"))
    if not latest_files:
        return 0, 0, 0.0

    checkpoint_path = max(
        latest_files, key=lambda path: int(path.rsplit("_e", 1)[1].split(".")[0])
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    logger.info("Resumed from {}", checkpoint_path)
    return (
        checkpoint["Epoch"] + 1,
        checkpoint.get("total_steps", 0),
        checkpoint.get("best_AUC", 0.0),
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, total_steps, best_auc):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "Epoch": epoch,
            "total_steps": total_steps,
            "best_AUC": best_auc,
        },
        path,
    )


def remove_other_checkpoints(run_dir, pattern, keep_path):
    for path in glob.glob(os.path.join(run_dir, pattern)):
        if path != keep_path:
            os.remove(path)


def train(options, config):
    if options.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(options.device)

    run_dir = os.path.join(options.save_dir, options.name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "running_args.json"), "w") as args_file:
        json.dump(vars(options), args_file, indent=2)
    writer = SummaryWriter(run_dir)

    train_dataset = MIPA(
        "train",
        options.setting,
        data_root=options.data_root,
        split_root=options.split_root,
        image_count=options.train_img_nums,
    )
    val_dataset = MIPA(
        "val",
        options.setting,
        data_root=options.data_root,
        split_root=options.split_root,
        image_count=options.train_img_nums,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        shuffle=False,
    )
    logger.info(
        "Setting={} variant={} images={} IAM layers={}",
        options.setting,
        options.variant,
        options.train_img_nums,
        options.invariant_extract_layers,
    )

    model = build_model(options, config, pre_train=True).to(device)
    heatmap_loss = HM_Loss().to(device)
    classification_loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=options.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=options.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=options.epochs, eta_min=1e-6
    )

    start_epoch, total_steps, best_auc = 0, 0, 0.0
    if options.resume:
        start_epoch, total_steps, best_auc = restore_latest(
            run_dir, model, optimizer, scheduler, device
        )

    for epoch in range(start_epoch, options.epochs):
        model.train()
        epoch_loss = 0.0
        for iteration, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, components = model_loss(
                model,
                batch,
                device,
                heatmap_loss,
                classification_loss,
                options,
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("train/loss", loss.item(), total_steps)
            for name, value in components.items():
                writer.add_scalar(f"train/{name}_loss", value.item(), total_steps)
            if iteration % 50 == 0:
                logger.info(
                    "Epoch {} iteration {} loss {:.6f}", epoch, iteration, loss.item()
                )
            total_steps += 1

        mean_loss = epoch_loss / len(train_loader)
        writer.add_scalar("train/epoch_loss", mean_loss, epoch)
        logger.info("Epoch {} mean loss {:.6f}", epoch, mean_loss)

        metrics = evaluate(model, val_loader, device)
        logger.info(
            "AUC:{AUC:.6f} | aIOU:{aIOU:.6f} | SIM:{SIM:.6f} | MAE:{MAE:.6f}",
            **metrics,
        )
        for name, value in metrics.items():
            writer.add_scalar(f"val/{name}", value, epoch)

        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            best_path = os.path.join(run_dir, f"best_e{epoch}.pt")
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, total_steps, best_auc
            )
            remove_other_checkpoints(run_dir, "best_e*.pt", best_path)
            logger.info("Best model saved at {}", best_path)

        latest_path = os.path.join(run_dir, f"latest_e{epoch}.pt")
        save_checkpoint(
            latest_path, model, optimizer, scheduler, epoch, total_steps, best_auc
        )
        remove_other_checkpoints(run_dir, "latest_e*.pt", latest_path)
        scheduler.step()

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="configs/mifag_seen.yaml")
    parser.add_argument("--save_dir", default="runs/train")
    parser.add_argument("--name", default="mifag")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--data_root", default=None)
    parser.add_argument("--split_root", default=None)
    parser.add_argument("--resnet18_path", default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument(
        "--variant",
        choices=["full", "baseline", "iam_only", "adm_only"],
        default=None,
        help="Full MIFAG or a Table 2 ablation.",
    )
    parser.add_argument("--train_img_nums", type=int, default=None)
    parser.add_argument("--invariant_extract_layers", type=int, default=None)
    parser.add_argument("--droprate", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--loss_cls", type=float, default=None)
    parser.add_argument("--loss_sim", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything(42)
    cli_options = parse_args()
    yaml_config = read_config(cli_options.yaml)
    resolved_options = resolve_options(cli_options, yaml_config)
    train(resolved_options, yaml_config)
