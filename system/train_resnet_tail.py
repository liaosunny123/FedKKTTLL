import argparse
import copy
import json
import math
import os
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

from flcore.trainmodel.models_fedext import FedEXTModel


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone training for FedEXT classifier tail")
    parser.add_argument("--dataset-dir", required=True, help="Path to clients-feature dataset directory")
    parser.add_argument("--model", default="torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)", help="Model expression evaluated to build base encoder")
    parser.add_argument("--encoder-ratio", type=float, default=float(os.getenv("ENCODER_RATIO", 0.7)), help="Encoder ratio used to split the model")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-balanced-test", action="store_true", help="Evaluate on balanced test split if available")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=None, help="Number of classes (auto-detected if omitted)")
    parser.add_argument("--save-path", default=None, help="Optional path to save trained classifier")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="fedktl")
    parser.add_argument("--wandb-entity", default="epicmo")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--force-linear-projection", action="store_true",
                        help="当尾部需要卷积输入时，通过线性层映射维度而非回退到 head")
    return parser.parse_args()


def load_feature_dataset(dataset_dir, filename):
    path = os.path.join(dataset_dir, filename)
    if not os.path.exists(path):
        return None, None
    payload = torch.load(path, map_location="cpu")
    return payload["features"], payload["labels"]


def build_classifier(
    model_expr,
    encoder_ratio,
    input_feature_dim,
    model_feature_dim,
    num_classes,
    device,
    dataset_is_flat,
    input_shape,
    original_feature_shape,
    force_linear_projection,
):
    dummy_args = SimpleNamespace(
        models=[model_expr],
        num_classes=num_classes,
        feature_dim=model_feature_dim,
        encoder_ratio=encoder_ratio,
    )
    fedext_model = FedEXTModel(dummy_args, 0)

    original_local_layers = [name for name, _ in (fedext_model.local_layers or [])]
    original_split_index = getattr(fedext_model, "layer_split_index", None)
    total_layers = len(fedext_model.layers_list or [])

    # Determine whether tail expects spatial (conv) input
    projection_added = False
    fallback_applied = False

    def is_head_layer(name: str) -> bool:
        return name.startswith("head")

    first_local_requires_spatial = False
    unresolved_spatial_tail = False
    if fedext_model.local_layers:
        first_local_name, _ = fedext_model.local_layers[0]
        first_local_requires_spatial = not is_head_layer(first_local_name)

    can_reconstruct_shape = original_feature_shape is not None and len(original_feature_shape) > 1
    if first_local_requires_spatial and not can_reconstruct_shape:
        unresolved_spatial_tail = True

    tail_requires_spatial = first_local_requires_spatial and can_reconstruct_shape

    if unresolved_spatial_tail:
        if total_layers > 0:
            fedext_model.set_layer_split(total_layers - 1)
            fallback_applied = True
        tail_requires_spatial = False

    if tail_requires_spatial and dataset_is_flat:
        if force_linear_projection and original_feature_shape is not None:
            projection_added = True
        else:
            if total_layers > 0:
                fedext_model.set_layer_split(total_layers - 1)
                fallback_applied = True
            tail_requires_spatial = False

    final_split_index = getattr(fedext_model, "layer_split_index", None)
    final_local_layers = [name for name, _ in (fedext_model.local_layers or [])]

    local_modules = []
    if projection_added:
        target_dim = int(math.prod(original_feature_shape))
        projection = nn.Sequential(
            nn.Linear(input_feature_dim, target_dim),
            nn.Unflatten(1, original_feature_shape),
        )
        local_modules.append(projection)

    for _, layer in fedext_model.local_layers:
        local_modules.append(copy.deepcopy(layer))

    if len(local_modules) == 0:
        raise ValueError("No local layers found after splitting; check encoder_ratio")

    classifier = nn.Sequential(*local_modules) if len(local_modules) > 1 else local_modules[0]
    classifier = classifier.to(device)

    if projection_added:
        tail_type = "conv_with_projection"
    elif final_local_layers and not is_head_layer(final_local_layers[0]):
        tail_type = "conv_native"
    else:
        tail_type = "head"

    info = {
        "original_local_layers": original_local_layers,
        "final_local_layers": final_local_layers,
        "fallback_applied": fallback_applied,
        "projection_added": projection_added,
        "original_split_index": original_split_index,
        "final_split_index": final_split_index,
        "total_layers": total_layers,
        "tail_type": tail_type,
        "dataset_is_flat": dataset_is_flat,
        "input_shape": input_shape,
        "original_feature_shape": tuple(original_feature_shape) if original_feature_shape else None,
        "model_feature_dim": model_feature_dim,
        "input_feature_dim": input_feature_dim,
        "requires_flatten_input": False,
        "requires_reshape_input": False,
        "projection_shape": tuple(original_feature_shape) if projection_added else None,
    }

    if tail_type in ("head", "conv_with_projection") and not dataset_is_flat:
        info["requires_flatten_input"] = True
    if tail_type == "conv_native" and dataset_is_flat and original_feature_shape is not None:
        info["requires_reshape_input"] = True

    return classifier, info


def create_dataloaders(train_features, train_labels, batch_size):
    train_dataset = TensorDataset(train_features, train_labels)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def prepare_features_for_classifier(features, info, device):
    feats = features.to(device)
    tail_type = info.get("tail_type", "head")
    expected_shape = info.get("original_feature_shape")

    if tail_type in ("head", "conv_with_projection"):
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
    elif tail_type == "conv_native":
        if feats.ndim == 2 and expected_shape is not None:
            feats = feats.reshape(feats.size(0), *expected_shape)

    return feats


def evaluate(model, features, labels, device, info):
    model.eval()
    with torch.no_grad():
        inputs = prepare_features_for_classifier(features, info, device)
        targets = labels.to(device)
        outputs = model(inputs)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == targets).float().mean().item()
    return accuracy


def main():
    args = parse_args()

    metadata_path = os.path.join(args.dataset_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {args.dataset_dir}")

    with open(metadata_path, "r", encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)

    train_features, train_labels = load_feature_dataset(args.dataset_dir, "global_train_dataset.pt")
    if train_features is None:
        raise FileNotFoundError("global_train_dataset.pt not found, cannot proceed")

    if args.use_balanced_test:
        test_features, test_labels = load_feature_dataset(args.dataset_dir, "global_test_balanced_dataset.pt")
        if test_features is None:
            raise FileNotFoundError("Balanced test set requested but not available")
    else:
        test_features, test_labels = load_feature_dataset(args.dataset_dir, "global_test_dataset.pt")
        if test_features is None:
            raise FileNotFoundError("global_test_dataset.pt not found, cannot proceed")

    original_feature_shape = metadata.get("train_feature_shape_before_flatten")
    if original_feature_shape:
        original_feature_shape = tuple(original_feature_shape)

    model_feature_dim_meta = metadata.get("model_feature_dim")

    num_classes = args.num_classes or int(train_labels.max().item() + 1)

    train_features = train_features.float()
    train_labels = train_labels.long()
    test_features = test_features.float()
    test_labels = test_labels.long()

    input_shape = tuple(train_features.shape[1:]) if train_features.ndim > 1 else ()
    if original_feature_shape is None and len(input_shape) > 1:
        original_feature_shape = input_shape

    dataset_is_flat = len(input_shape) <= 1
    if input_shape:
        input_feature_dim = int(math.prod(input_shape))
    else:
        input_feature_dim = int(train_features.reshape(train_features.size(0), -1).shape[1])

    model_feature_dim = (
        int(model_feature_dim_meta)
        if model_feature_dim_meta is not None
        else input_feature_dim
    )

    device = torch.device(args.device)
    classifier, classifier_info = build_classifier(
        model_expr=args.model,
        encoder_ratio=args.encoder_ratio,
        input_feature_dim=input_feature_dim,
        model_feature_dim=model_feature_dim,
        num_classes=num_classes,
        device=device,
        dataset_is_flat=dataset_is_flat,
        input_shape=input_shape,
        original_feature_shape=original_feature_shape,
        force_linear_projection=args.force_linear_projection,
    )

    tail_param_count = sum(p.numel() for p in classifier.parameters())
    input_shape_for_log = list(input_shape) if input_shape else []
    original_shape_for_log = list(original_feature_shape) if original_feature_shape else None
    projection_shape_for_log = (
        list(classifier_info.get("projection_shape"))
        if classifier_info.get("projection_shape")
        else None
    )
    print("\n=== 尾部模型信息 ===")
    tail_type = classifier_info["tail_type"]
    if classifier_info["fallback_applied"]:
        print("检测到原始切分包含卷积层，已自动回退为仅训练分类头。")
    elif tail_type == "conv_with_projection":
        proj_shape = classifier_info.get("projection_shape")
        proj_shape_str = f" -> {proj_shape}" if proj_shape else ""
        print(f"使用线性映射适配尾部卷积层{proj_shape_str}。")
    elif tail_type == "conv_native":
        print("使用卷积尾部直接训练。")
    else:
        print("使用线性分类头。")
    if classifier_info["original_local_layers"]:
        print("原始尾部层:")
        for name in classifier_info["original_local_layers"]:
            print(f"  - {name}")
    if classifier_info["final_local_layers"]:
        print("实际参与训练的尾部层:")
        for name in classifier_info["final_local_layers"]:
            print(f"  - {name}")
    if args.force_linear_projection and not classifier_info["projection_added"] and tail_type == "head":
        print("提示: 已开启线性映射选项，但由于缺少卷积特征形状信息，仍退回到线性分类头。")
    print(f"尾部总参数量: {tail_param_count:,}")
    total_layers = classifier_info["total_layers"]
    orig_index = classifier_info["original_split_index"]
    final_index = classifier_info["final_split_index"]
    if total_layers:
        if orig_index is not None:
            print(f"原始切分位置: {orig_index}/{total_layers} (比例约 {orig_index/total_layers:.2f})")
        if final_index is not None and final_index != orig_index:
            print(f"实际切分位置: {final_index}/{total_layers} (比例约 {final_index/total_layers:.2f})")
    shape_str = str(input_shape_for_log) if input_shape_for_log else "[flattened]"
    print(f"输入特征形状: {shape_str} (扁平存储: {'是' if dataset_is_flat else '否'})")
    if classifier_info.get("requires_flatten_input"):
        print("输入特征将自动展平成二维后再送入尾部模型。")
    if classifier_info.get("requires_reshape_input") and classifier_info.get("original_feature_shape"):
        print(f"输入特征将在训练/评估前重塑为 {classifier_info['original_feature_shape']}。")

    sample_input = prepare_features_for_classifier(train_features[:2], classifier_info, device)
    try:
        _ = classifier(sample_input)
    except RuntimeError as err:
        raise RuntimeError("Classifier forward pass failed. Ensure encoder_ratio matches dataset features") from err

    optimizer = optim.SGD(
        classifier.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but --use-wandb was provided")
        run_name = args.wandb_run_name or f"standalone-resnet-tail-{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config={
                "encoder_ratio": args.encoder_ratio,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "num_classes": num_classes,
                "model_feature_dim": model_feature_dim,
                "input_feature_dim": input_feature_dim,
                "train_samples": int(train_features.size(0)),
                "test_samples": int(test_features.size(0)),
                "dataset_dir": args.dataset_dir,
                "model": args.model,
                "tail_param_count": tail_param_count,
                "tail_layers": classifier_info["final_local_layers"],
                "tail_fallback": classifier_info["fallback_applied"],
                "tail_original_layers": classifier_info["original_local_layers"],
                "tail_original_split_index": classifier_info["original_split_index"],
                "tail_final_split_index": classifier_info["final_split_index"],
                "tail_type": classifier_info["tail_type"],
                "tail_projection_added": classifier_info["projection_added"],
                "tail_projection_shape": projection_shape_for_log,
                "tail_requires_flatten_input": classifier_info["requires_flatten_input"],
                "tail_requires_reshape_input": classifier_info["requires_reshape_input"],
                "dataset_is_flat": dataset_is_flat,
                "input_feature_shape": input_shape_for_log,
                "original_feature_shape": original_shape_for_log,
                "force_linear_projection": args.force_linear_projection,
            },
        )
        wandb.define_metric("Server/step")
        wandb.define_metric("Server/classifier_train_loss", step_metric="Server/step")
        wandb.define_metric("Server/classifier_train_accuracy", step_metric="Server/step")
        wandb.define_metric("Server/global_model_test_accuracy", step_metric="Server/step")

    train_loader = create_dataloaders(train_features, train_labels, args.batch_size)

    for epoch in range(args.epochs):
        classifier.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features = prepare_features_for_classifier(features, classifier_info, device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / total
        train_acc = correct / total

        if args.use_wandb:
            wandb.log(
                {
                    "Server/step": epoch,
                    "Server/classifier_epoch": epoch,
                    "Server/classifier_train_loss": avg_loss,
                    "Server/classifier_train_accuracy": train_acc,
                    "Server/classifier_learning_rate": args.learning_rate,
                }
            )

        print(f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, acc={train_acc:.4f}")

    test_accuracy = evaluate(classifier, test_features, test_labels, device, classifier_info)

    print(f"Test accuracy: {test_accuracy:.4f} on {test_labels.size(0)} samples")

    if args.use_wandb:
        wandb.log(
            {
                "Server/step": args.epochs,
                "Server/global_model_test_accuracy": test_accuracy,
                "Server/global_model_test_samples": int(test_labels.size(0)),
                "Server/global_model_train_samples": int(train_labels.size(0)),
                "Server/balanced_samples_per_client": metadata.get("balanced_samples_per_client"),
            }
        )

    if args.save_path:
        torch.save({"classifier_state_dict": classifier.state_dict()}, args.save_path)
        print(f"Saved classifier to {args.save_path}")


if __name__ == "__main__":
    main()
