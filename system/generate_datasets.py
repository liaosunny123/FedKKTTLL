import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from utils.data_distribution import DataDistributionManager
from utils.data_utils import read_client_data


def parse_args():
    parser = argparse.ArgumentParser(description="依据保存的客户端模型生成特征/标签数据集")
    parser.add_argument("--run-dir", required=True, help="保存客户端模型的目录 (如 temp/Cifar10/FedEXT/1680000000.0)")
    parser.add_argument("--dataset", required=True, help="数据集名称 (如 Cifar10)")
    parser.add_argument("--num-clients", type=int, default=20, help="客户端数量")
    parser.add_argument("--batch-size", type=int, default=128, help="特征提取批次大小")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="模型推理设备")
    parser.add_argument("--distribution-config", default="", help="可选：数据分布配置 JSON 文件路径")
    parser.add_argument("--encoder-ratio", type=float, default=None, help="覆盖模型内记录的 encoder 切分比例")
    parser.add_argument("--num-classes", type=int, default=10, help="类别数量")
    parser.add_argument("--output-dir", default=None, help="输出目录，默认为 run-dir/clients-feature")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，用于平衡测试集采样")
    return parser.parse_args()


def extract_features(model, dataloader, device):
    features_list = []
    labels_list = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            if isinstance(x, list):
                for idx, item in enumerate(x):
                    if torch.is_tensor(item):
                        x[idx] = item.to(device)
                input_data = x
            elif isinstance(x, tuple):
                input_data = tuple(item.to(device) if torch.is_tensor(item) else item for item in x)
            else:
                input_data = x.to(device)
            y = y.to(device)

            if hasattr(model, "forward_split"):
                feat, _ = model.forward_split(input_data)
            else:
                feat = model.extract_global_features(input_data)

            if isinstance(feat, tuple):
                feat = feat[0]

            if len(feat.shape) > 2:
                feat = torch.flatten(feat, 1)

            features_list.append(feat.cpu())
            labels_list.append(y.cpu())

    if not features_list:
        return torch.empty((0, model.feature_dim if hasattr(model, "feature_dim") else 0)), torch.empty((0,), dtype=torch.long)

    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    return features, labels


def load_client_model(run_dir, client_id, device):
    path = os.path.join(run_dir, f"Client_{client_id}_model.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到客户端 {client_id} 模型文件: {path}")
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model.to(device)
    model.eval()
    return model


def prepare_dataloader(dataset_name, client_id, batch_size, distribution_manager, num_classes, is_train=True):
    raw_data = read_client_data(dataset_name, client_id, is_train=is_train)
    if distribution_manager and distribution_manager.config:
        raw_data = distribution_manager.filter_client_data(client_id, raw_data, num_classes, is_train=is_train)
    return DataLoader(raw_data, batch_size=batch_size, shuffle=False, drop_last=False)


def save_feature_datasets(
    output_dir,
    source_run_dir,
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    balanced_test_embeddings,
    balanced_test_labels,
    train_features_per_client,
    train_labels_per_client,
    test_features_per_client,
    test_labels_per_client,
    balanced_samples_per_client,
    client_ids,
    extra_metadata=None,
):
    os.makedirs(output_dir, exist_ok=True)

    torch.save({"features": train_embeddings.cpu(), "labels": train_labels.cpu()}, os.path.join(output_dir, "global_train_dataset.pt"))
    torch.save({"features": test_embeddings.cpu(), "labels": test_labels.cpu()}, os.path.join(output_dir, "global_test_dataset.pt"))

    if balanced_test_embeddings is not None and balanced_test_labels is not None:
        torch.save(
            {"features": balanced_test_embeddings.cpu(), "labels": balanced_test_labels.cpu()},
            os.path.join(output_dir, "global_test_balanced_dataset.pt"),
        )

    per_client_dir = os.path.join(output_dir, "per-client")
    os.makedirs(per_client_dir, exist_ok=True)

    for cid, train_feat, train_lab, test_feat, test_lab in zip(
        client_ids,
        train_features_per_client,
        train_labels_per_client,
        test_features_per_client,
        test_labels_per_client,
    ):
        torch.save(
            {
                "client_id": cid,
                "train_features": train_feat.cpu(),
                "train_labels": train_lab.cpu(),
                "test_features": test_feat.cpu(),
                "test_labels": test_lab.cpu(),
            },
            os.path.join(per_client_dir, f"client_{cid:03d}.pt"),
        )

    metadata = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "client_ids": list(client_ids),
        "train_samples": int(train_embeddings.shape[0]),
        "train_feature_dim": int(train_embeddings.shape[1]) if train_embeddings.ndim > 1 else 1,
        "test_samples": int(test_embeddings.shape[0]),
        "test_feature_dim": int(test_embeddings.shape[1]) if test_embeddings.ndim > 1 else 1,
        "balanced_test_samples": int(balanced_test_embeddings.shape[0]) if balanced_test_embeddings is not None else None,
        "balanced_test_feature_dim": int(balanced_test_embeddings.shape[1]) if balanced_test_embeddings is not None and balanced_test_embeddings.ndim > 1 else None,
        "balanced_samples_per_client": int(balanced_samples_per_client) if balanced_samples_per_client is not None else None,
        "train_samples_per_client": [int(feat.shape[0]) for feat in train_features_per_client],
        "test_samples_per_client": [int(feat.shape[0]) for feat in test_features_per_client],
        "source_run_dir": source_run_dir,
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"✅ 已保存特征数据集到: {output_dir}")


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(args.run_dir, "clients-feature")
    device = torch.device(args.device)

    distribution_manager = DataDistributionManager(args.distribution_config) if args.distribution_config else None

    generator = torch.Generator().manual_seed(args.seed)

    client_ids = list(range(args.num_clients))

    train_features_per_client = []
    train_labels_per_client = []
    test_features_per_client = []
    test_labels_per_client = []

    for cid in client_ids:
        print(f"\n====== 处理客户端 {cid} ======")
        model = load_client_model(args.run_dir, cid, device)

        if args.encoder_ratio is not None and hasattr(model, "update_split_ratio"):
            model.update_split_ratio(args.encoder_ratio)
        elif hasattr(model, "layer_split_index") and hasattr(model, "layers_list"):
            total_layers = len(getattr(model, "layers_list", []))
            current_split = getattr(model, "layer_split_index", None)
            if current_split is not None and total_layers:
                ratio = current_split / total_layers
                print(f"  使用模型内保存的切分: index={current_split}/{total_layers}, ratio≈{ratio:.2f}")
        elif hasattr(model, "get_split_info"):
            info = model.get_split_info()
            split_index = info.get("split_index")
            total_layers = info.get("total_layers")
            encoder_ratio = info.get("encoder_ratio")
            print(f"  使用模型内保存的切分: index={split_index}/{total_layers}, ratio≈{encoder_ratio:.2f}")

        train_loader = prepare_dataloader(args.dataset, cid, args.batch_size, distribution_manager, args.num_classes, is_train=True)
        train_feat, train_lab = extract_features(model, train_loader, device)

        test_loader = prepare_dataloader(args.dataset, cid, args.batch_size, distribution_manager, args.num_classes, is_train=False)
        test_feat, test_lab = extract_features(model, test_loader, device)

        train_features_per_client.append(train_feat)
        train_labels_per_client.append(train_lab)
        test_features_per_client.append(test_feat)
        test_labels_per_client.append(test_lab)

        print(f"  训练样本: {train_feat.shape[0]}, 测试样本: {test_feat.shape[0]}")

    train_embeddings = torch.cat(train_features_per_client, dim=0) if train_features_per_client else torch.empty(0)
    train_labels = torch.cat(train_labels_per_client, dim=0) if train_labels_per_client else torch.empty(0, dtype=torch.long)
    test_embeddings = torch.cat(test_features_per_client, dim=0) if test_features_per_client else torch.empty(0)
    test_labels = torch.cat(test_labels_per_client, dim=0) if test_labels_per_client else torch.empty(0, dtype=torch.long)

    min_test_samples = min((feat.shape[0] for feat in test_features_per_client if feat.shape[0] > 0), default=0)
    balanced_test_embeddings = None
    balanced_test_labels = None

    if min_test_samples > 0:
        balanced_emb_list = []
        balanced_lab_list = []
        for emb, lab in zip(test_features_per_client, test_labels_per_client):
            if emb.shape[0] < min_test_samples:
                raise ValueError("存在客户端测试样本不足，无法构建平衡测试集")
            indices = torch.randperm(emb.shape[0], generator=generator)[:min_test_samples]
            balanced_emb_list.append(emb[indices])
            balanced_lab_list.append(lab[indices])
        balanced_test_embeddings = torch.cat(balanced_emb_list, dim=0)
        balanced_test_labels = torch.cat(balanced_lab_list, dim=0)

    extra_metadata = {
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "encoder_ratio_override": args.encoder_ratio,
        "distribution_config": args.distribution_config or None,
        "batch_size": args.batch_size,
        "device": args.device,
        "seed": args.seed,
    }

    save_feature_datasets(
        output_dir,
        args.run_dir,
        train_embeddings,
        train_labels,
        test_embeddings,
        test_labels,
        balanced_test_embeddings,
        balanced_test_labels,
        train_features_per_client,
        train_labels_per_client,
        test_features_per_client,
        test_labels_per_client,
        min_test_samples if min_test_samples > 0 else None,
        client_ids,
        extra_metadata,
    )


if __name__ == "__main__":
    main()
