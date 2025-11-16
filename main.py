import os



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devices = [0]
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchio
from torchio.transforms import ZNormalization
from tqdm import tqdm
import pandas as pd
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir

source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

source_val_dir = hp.source_val_dir
label_val_dir = hp.label_val_dir

output_dir_test = hp.output_dir_test


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-o', '--log_dir', type=str, default=hp.log_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')
    parser.add_argument('--best_dice_model_file', type=str, default=hp.best_dice_model_file,
                        help='Store the best_dice_model checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    training.add_argument('--best_dice', type=int, default=hp.best_dice, help='best-dice')
    parser.add_argument('-k', "--ckpt", type=str, default=hp.ckpt, help="path to the checkpoints to resume training")
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')
    return parser


import os
import torch
from torchvision.utils import save_image  # 需提前安装 torchvision


def _min_max_norm(t: torch.Tensor) -> torch.Tensor:
    """将张量线性归一化到 [0,1] 区间，避免显示全黑/全白。"""
    t_min, t_max = t.min(), t.max()
    return (t - t_min) / (t_max - t_min + 1e-8)


def validate(epoch, model, val_loader, criterion, hp, metric, device: str = "cuda", save_outputs: bool = False):
    """
    Run one full pass on the validation set.

    Returns
    -------
    dict: Dictionary containing the averaged metrics:
        {
            'loss', 'dice', 'iou', 'precision', 'recall', 'FPR', 'FNR', 'acc', 'tp', 'fp', 'fn', 'tn'
        }
    """
    model.eval()

    totals = {
        "loss": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "FPR": 0.0,
        "FNR": 0.0,
        "acc": 0.0,
        "tp": 0.0,
        "fp": 0.0,
        "fn": 0.0,
        "tn": 0.0,
        "specificity": 0.0,
        "NPV": 0.0,
        "f1": 0.0,
        "balanced_acc": 0.0,
        "MCC": 0.0,
    }
    num_iters = 0

    # ── 创建输出目录 ─────────────────────────────────────────────────────────────
    if save_outputs:
        images_dir = os.path.join(hp.output_dir_test, os.path.join(str(epoch),"images"))
        masks_dir = os.path.join(hp.output_dir_test, os.path.join(str(epoch),"masks"))
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if getattr(hp, "debug", False) and i >= 1:  # 可选调试
                break

            x = batch["source"]["data"].float().to(device)  # (B, C, D, H, W) 或 (B, C, H, W)
            y = batch["label"]["data"].float().to(device)

            if hp.mode == "2d":
                x = x.squeeze(4)  # → (B, C, H, W)
                y = y.squeeze(4)
                y[y != 0] = 1  # 二值化标签
                if x.shape[1] == 1:  # 假设通道维度在 dim=1
                    x = x.repeat(1, 3, 1, 1)

            outputs = model(x)
            loss = criterion(outputs, y)

            logits = torch.sigmoid(outputs)
            labels = (logits > 0.5).float()  # 预测掩膜 (B, 1, H, W)

            # ── 统计所有指标 ───────────────────────────────────────────────────────
            metrics = metric(y.cpu(), labels.cpu())  # 获取所有指标字典

            # 累加各指标值
            totals["loss"] += loss.item()
            totals["dice"] += metrics["dice"]
            totals["iou"] += metrics["iou"]
            totals["precision"] += metrics["precision"]
            totals["recall"] += metrics["recall"]
            totals["FPR"] += metrics["FPR"]
            totals["FNR"] += metrics["FNR"]
            totals["acc"] += metrics["acc"]
            totals["tp"] += metrics["tp"]
            totals["fp"] += metrics["fp"]
            totals["fn"] += metrics["fn"]
            totals["tn"] += metrics["tn"]
            totals["specificity"] += metrics["specificity"]
            totals["NPV"] += metrics["NPV"]
            totals["f1"] += metrics["f1"]
            totals["balanced_acc"] += metrics["balanced_acc"]
            totals["MCC"] += metrics["MCC"]
            num_iters += 1

            # ── 可选保存输出 ───────────────────────────────────────────────────
            if save_outputs:
                x_cpu = x.cpu()
                labels_cpu = labels.cpu()
                B = x_cpu.size(0)
                for b in range(B):
                    # 归一化输入，保持灰度 / RGB 不变
                    img_to_save = _min_max_norm(x_cpu[b])
                    mask_to_save = labels_cpu[b]

                    filename = os.path.basename(batch["source"]["path"][b])
                    img_path = os.path.join(images_dir, filename)
                    mask_path = os.path.join(masks_dir, filename)

                    # 若 C==1，save_image 会自动将 1-通道张量保存为灰度
                    save_image(img_to_save, img_path)
                    save_image(mask_to_save, mask_path)

    # ── 取平均 ──────────────────────────────────────────────────────────────────
    for k in totals:
        totals[k] /= num_iters

    # 返回所有计算后的指标，包括准确率和其他详细统计
    return totals


import os
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import argparse


def train(model, optimizer):
    parser = argparse.ArgumentParser(description='PyTorch Image Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    os.makedirs(args.log_dir, exist_ok=True)

    # 修正: devicess -> devices
    model = torch.nn.DataParallel(model, device_ids=devices)

    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.log_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.log_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    best_dice = getattr(args, "best_dice", float("-inf"))
    model.cuda()
    writer = SummaryWriter(args.log_dir)
    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)

    best_metrics = {
        "epoch": None,
        "val_loss": None,
        "val_dice": None,
        "val_iou": None,
        "val_precision": None,
        "val_recall": None,
        "val_FPR": None,
        "val_FNR": None,
        "val_acc": None,
        "val_specificity": None,
        "val_NPV": None,
        "val_f1": None,
        "val_balanced_acc": None,
        "val_MCC": None,
        "ckpt_path": None,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch += elapsed_epochs
        num_iters = 0
        for i, batch in enumerate(train_loader):

            optimizer.zero_grad()

            if (hp.in_class == 1 and hp.out_class == 1) or (hp.in_class == 3 and hp.out_class == 1):
                x = batch['source']['data']
                y = batch['label']['data']

                x = x.type(torch.FloatTensor).cuda()
                y = y.type(torch.FloatTensor).cuda()

            else:
                x = batch['source']['data']
                y_atery = batch['atery']['data']
                y_lung = batch['lung']['data']
                y_trachea = batch['trachea']['data']
                y_vein = batch['vein']['data']

                x = x.type(torch.FloatTensor).cuda()
                y = torch.cat((y_atery, y_lung, y_trachea, y_vein), 1)
                y = y.type(torch.FloatTensor).cuda()

            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)
                y[y != 0] = 1
                if x.shape[1] == 1:  # 假设通道维度在 dim=1
                    x = x.repeat(1, 3, 1, 1)

            outputs = model(x)

            logits = torch.sigmoid(outputs)
            labels = logits.clone()
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0

            loss = criterion(outputs, y)

            num_iters += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            iteration += 1

            # 获取所有的指标
            m = metric(y.cpu(), labels.cpu())

            dice = m["dice"]
            iou = m["iou"]
            precision = m["precision"]
            recall = m["recall"]
            FPR = m["FPR"]
            FNR = m["FNR"]
            acc = m["acc"]
            specificity = m["specificity"]
            npv = m["NPV"]
            f1 = m["f1"]
            balanced_acc = m["balanced_acc"]
            mcc = m["MCC"]
            tp = m["tp"]
            fp = m["fp"]
            fn = m["fn"]
            tn = m["tn"]

            # 将指标写入 tensorboard
            writer.add_scalar('Training/Loss', loss.item(), iteration)
            writer.add_scalar('Training/FPR', FPR, iteration)
            writer.add_scalar('Training/FNR', FNR, iteration)
            writer.add_scalar('Training/dice', dice, iteration)
            writer.add_scalar('Training/iou', iou, iteration)
            writer.add_scalar('Training/precision', precision, iteration)
            writer.add_scalar('Training/recall', recall, iteration)
            writer.add_scalar('Training/acc', acc, iteration)
            writer.add_scalar('Training/specificity', specificity, iteration)
            writer.add_scalar('Training/NPV', npv, iteration)
            writer.add_scalar('Training/f1', f1, iteration)
            writer.add_scalar('Training/balanced_acc', balanced_acc, iteration)
            writer.add_scalar('Training/MCC', mcc, iteration)

        scheduler.step()

        # Store latest checkpoint in each epoch
        torch.save(
            {"model": model.state_dict(), "optim": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "epoch": epoch},
            os.path.join(args.log_dir, args.latest_checkpoint_file)
        )

        if epoch % args.epochs_per_checkpoint == 0:
            model.eval()
            val_metrics = validate(epoch, model, val_loader, criterion, hp, metric, device="cuda")

            print(f"[Epoch {epoch}] "
                  f"loss={val_metrics['loss']:.4f}, "
                  f"dice={val_metrics['dice']:.4f}, "
                  f"iou={val_metrics['iou']:.4f}")

            # 将验证结果写入 tensorboard
            writer.add_scalar('Validation/val_loss', val_metrics["loss"], epoch)
            writer.add_scalar('Validation/val_dice', val_metrics["dice"], epoch)
            writer.add_scalar('Validation/val_iou', val_metrics["iou"], epoch)
            writer.add_scalar('Validation/val_precision', val_metrics["precision"], epoch)
            writer.add_scalar('Validation/val_recall', val_metrics["recall"], epoch)
            writer.add_scalar('Validation/val_FNR', val_metrics["FNR"], epoch)
            writer.add_scalar('Validation/val_FPR', val_metrics["FPR"], epoch)
            writer.add_scalar('Validation/val_acc', val_metrics["acc"], epoch)
            writer.add_scalar('Validation/val_specificity', val_metrics["specificity"], epoch)
            writer.add_scalar('Validation/val_NPV', val_metrics["NPV"], epoch)
            writer.add_scalar('Validation/val_f1', val_metrics["f1"], epoch)
            writer.add_scalar('Validation/val_balanced_acc', val_metrics["balanced_acc"], epoch)
            writer.add_scalar('Validation/val_MCC', val_metrics["MCC"], epoch)

            # 如果更好，保存 best，并记录“最好epoch的所有指标”
            if val_metrics["dice"] > best_dice:
                print(f"Dice improved from {best_dice:.4f} to {val_metrics['dice']:.4f}. Saving best model...")
                best_dice = val_metrics["dice"]
                best_path = os.path.join(args.log_dir, "best_dice_model.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    best_path,
                )
                best_metrics.update({
                    "epoch": epoch,
                    "val_loss": float(val_metrics["loss"]),
                    "val_dice": float(val_metrics["dice"]),
                    "val_iou": float(val_metrics["iou"]),
                    "val_precision": float(val_metrics["precision"]),
                    "val_recall": float(val_metrics["recall"]),
                    "val_FPR": float(val_metrics["FPR"]),
                    "val_FNR": float(val_metrics["FNR"]),
                    "val_acc": float(val_metrics["acc"]),
                    "val_specificity": float(val_metrics["specificity"]),
                    "val_NPV": float(val_metrics["NPV"]),
                    "val_f1": float(val_metrics["f1"]),
                    "val_balanced_acc": float(val_metrics["balanced_acc"]),
                    "val_MCC": float(val_metrics["MCC"]),

                    # TP/FP/FN/TN 也保存（你后面分析可能需要）
                    "val_tp": float(val_metrics["tp"]),
                    "val_fp": float(val_metrics["fp"]),
                    "val_fn": float(val_metrics["fn"]),
                    "val_tn": float(val_metrics["tn"]),

                    "ckpt_path": best_path,
                })

    writer.close()

    return best_metrics


def test(model):
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    os.makedirs(output_dir_test, exist_ok=True)

    model = torch.nn.DataParallel(model, device_ids=devices)

    print("load model:", args.ckpt)
    print(os.path.join(args.log_dir, args.best_dice_model_file))
    ckpt = torch.load(os.path.join(args.log_dir, args.best_dice_model_file),
                      map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])

    model.cuda()

    znorm = ZNormalization()

    if hp.mode == '3d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size
    elif hp.mode == '2d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size

    dice_scores = []
    Pids = []

    for i, subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
            subj,
            patch_size,
            patch_overlap,
        )
        print(test_dataset.image_paths[i])
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=args.batch)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)

        model.eval()

        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):

                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                if hp.mode == '2d':
                    input_tensor = input_tensor.squeeze(4)
                outputs = model(input_tensor)

                if hp.mode == '2d':
                    outputs = outputs.unsqueeze(4)
                logits = torch.sigmoid(outputs)

                labels = logits.clone()
                labels[labels > 0.5] = 1
                labels[labels <= 0.5] = 0
                aggregator.add_batch(logits, locations)
                aggregator_1.add_batch(labels, locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()
        affine = subj['source']['affine']

        # dice = metric(subj['label'][torchio.DATA].to(device), output_tensor_1.to(device))

        dice = metric(subj['label'][torchio.DATA].to('cpu'), output_tensor_1.to('cpu'))
        print('校验', subj['label'][torchio.DATA].shape, output_tensor_1.shape)
        dice_scores.append(dice[0].item())
        Pid = os.path.basename(test_dataset.image_paths[i])
        Pids.append(Pid)
        print(f"Dice Score for sample {i}: {dice[0].item():.4f}")

        if (hp.out_class == 1):
            output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy() * 255, affine=affine)
            print(output_tensor_1.numpy().shape)

            output_image.save(os.path.join(output_dir_test, os.path.basename(test_dataset.image_paths[i])))

    df = pd.DataFrame({
        "ID": Pids,  # 样本 ID 或文件名
        "Dice Score": dice_scores  # Dice 系数
    })
    df.to_csv(os.path.join(output_dir_test, "dice_scores.csv"), index=False)
    print("Dice scores saved to dice_scores.csv")

    # 打印平均 Dice 系数
    print("Average Dice Score:", sum(dice_scores) / len(dice_scores))


def set_seed(seed):
    """
    设置随机数种子，包括 CPU 和 GPU。
    """
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    torch.cuda.manual_seed(seed)  # 设置 GPU 的随机数种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
    torch.backends.cudnn.deterministic = True  # 确保结果的可重复性
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的优化，确保结果一致

import csv
import os

def save_metrics_to_csv(model_name, metrics: dict, csv_path: str):
    """
    保存训练指标到 CSV 文件。

    参数:
        model_name : str
            模型名字（写入 CSV 第一列）
        metrics : dict
            训练过程中返回的 best_metrics，例如：
            {
                "epoch": 10,
                "val_loss": 0.123,
                "val_dice": 0.876,
                ...
            }
        csv_path : str
            CSV 文件路径（需要包含 .csv 后缀）
    """

    # 确保目录存在
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # 表头和一行数据
    header = ["model_name"] + list(metrics.keys())
    row = [model_name] + list(metrics.values())

    # 如果文件不存在，先写表头
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

    print(f"[{model_name}] Metrics saved to {csv_path}")


if __name__ == '__main__':
    set_seed(42)
    from data_function import MedData_train, MedData_val
    from data_function import MedData_test

    train_dataset = MedData_train(source_train_dir, label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset, batch_size=hp.batch_size, shuffle=True,
                              num_workers=hp.num_workers,
                              pin_memory=False, drop_last=True)

    val_dataset = MedData_val(source_val_dir, label_val_dir)
    val_loader = DataLoader(val_dataset.queue_dataset, batch_size=hp.batch_size, shuffle=False,
                            num_workers=hp.num_workers,
                            pin_memory=False, drop_last=False)

    test_dataset = MedData_test(source_test_dir, label_test_dir)
    # model_names=['unet-dropout','Unet_DC_ED','Unet_SK_E','Unet_SK_ED','Unet_SVD_SH','Unet_SVD_beforeINC']
    # model_names=['Unet','Unet_DC_ED','Unet_SK_ED','Unet_SVD_SH','Unet_SKD','Unet_SKD_SVD']
    # model_names = ['AttentionUnet','UNETR','DenseNet121','UCTransNet','TransFuse','MISSFormer']
    # model_names = ['MISSFormer']
    model_names = [
    # "Unet",
    "DeepLabV3",
    # "miniseg",
    # "segnet",
    # "unetpp",
    # "SwinUNETR",
    # "AttentionUnet",
    # "UNETR",
    # "MISSFormer",
    # "UCTransNet",
    # "TransFuse"
]


    from loss_function import DiceLoss, Binary_Loss

    # criterion =  DiceLoss().cuda()
    criterion = Binary_Loss().cuda()
    # criterion =  FocalLoss().cuda()
    # criterion = DiceFocalLoss().cuda()
    # criterion = BCEDiceLoss().cuda()
    # criterion = BCEDiceFocalLoss().cuda()


    for model_name in model_names:
        print(model_name)
        if model_name == 'Unet':
            from models.two_d.unet import Unet
            model = Unet(in_channels=3, out_channels=1).to(device)
        if model_name == 'DeepLabV3':
            from models.two_d.deeplab import DeepLabV3
            model = DeepLabV3(in_channels=3, out_channels=1).to(device)
        if model_name == 'miniseg':
            from models.two_d.miniseg import MiniSeg
            model = MiniSeg(in_channels=3, out_channels=1).to(device)
        if model_name == 'segnet':
            from models.two_d.segnet import SegNet
            model = SegNet(in_channels=3, out_channels=1).to(device)
        if model_name == 'unetpp':
            from models.two_d.unetpp import ResNet34UnetPlus
            model = ResNet34UnetPlus(in_channels=3, out_channels=1).to(device)
        
        if model_name == 'SwinUNETR':
            from models.two_d.swin_unetr import SwinUNETR
            model = SwinUNETR(
            img_size=(256, 256),     # 仍然必填，长度与 spatial_dims 一致
            in_channels=3,
            out_channels=1,
            spatial_dims=2           # 声明 2‑D
        )
        if model_name == 'AttentionUnet':
            from models.two_d.attention_unet import AttentionUnet
            model = AttentionUnet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(32, 64, 128, 256),
            strides=(2, 2, 2),
            kernel_size=3,
            up_kernel_size=3,  # ← 改回奇数
            dropout=0.0,
        ).to(device)
        if model_name == 'UNETR':
            from models.two_d.UNETR import UNETR
            model = UNETR(
            in_channels=3,
            out_channels=1,        # 例如 2 类分割，可按需修改
            img_size=(256, 256),   # 与输入空间尺寸一致
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            proj_type="conv",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=2,        # ← 关键：2D
            qkv_bias=False,
            save_attn=False,
        ).to(device)

        if model_name == 'MISSFormer':
            from models.two_d.MISSFormer import MISSFormer
            model = MISSFormer(num_classes=1, token_mlp_mode="mix_skip")
            
        if model_name == 'UCTransNet':
            from models.two_d.UCTransNet import Cfg,UCTransNet
            cfg = Cfg() # img_size 设为 256，对应上面 patch_sizes 
            model = UCTransNet(cfg, n_channels=3, n_classes=1, img_size=256, vis=False)
        if model_name == 'TransFuse':
            from models.two_d.TransFuse import TransFuse_S,TransFuse_L, TransFuse_L_384
            model =TransFuse_S(num_classes=1, pretrained=False)
           

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.init_lr)
        hp.log_dir = os.path.join('logs', model_name )


       
        if hp.train_or_test == 'train':
            best_metrics = train(model, optimizer)
            csv_path = hp.source_train_dir+'ans.csv'
            save_metrics_to_csv(model_name,best_metrics,csv_path)

        
        elif hp.train_or_test == 'test':
            test(model)


