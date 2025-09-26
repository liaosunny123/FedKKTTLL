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

    projection_reason = None
    projection_target_shape = None

    def infer_conv_in_channels(module: nn.Module):
        if hasattr(module, "in_channels"):
            return getattr(module, "in_channels")
        if hasattr(module, "conv1") and hasattr(module.conv1, "in_channels"):
            return module.conv1.in_channels
        if isinstance(module, nn.Sequential):
            for child in module.children():
                result = infer_conv_in_channels(child)
                if result is not None:
                    return result
        for child in module.children():
            result = infer_conv_in_channels(child)
            if result is not None:
                return result
        return None

    if tail_requires_spatial and dataset_is_flat:
        if force_linear_projection and original_feature_shape is not None:
            projection_added = True
            projection_reason = "flat_dataset"
            projection_target_shape = original_feature_shape
        else:
            if total_layers > 0:
                fedext_model.set_layer_split(total_layers - 1)
                fallback_applied = True
            tail_requires_spatial = False
    elif tail_requires_spatial:
        first_local_module = fedext_model.local_layers[0][1] if fedext_model.local_layers else None
        expected_channels = infer_conv_in_channels(first_local_module) if first_local_module else None
        feature_channels = None
        if original_feature_shape and len(original_feature_shape) > 0:
            feature_channels = original_feature_shape[0]
        elif input_shape and len(input_shape) > 0:
            feature_channels = input_shape[0]

        if (
            expected_channels is not None
            and feature_channels is not None
            and expected_channels != feature_channels
        ):
            projection_added = True
            projection_reason = "channel_mismatch"
            spatial_dims = ()
            if original_feature_shape and len(original_feature_shape) > 1:
                spatial_dims = original_feature_shape[1:]
            elif input_shape and len(input_shape) > 1:
                spatial_dims = input_shape[1:]
            if not spatial_dims:
                spatial_dims = (1, 1)
            projection_target_shape = (expected_channels, *spatial_dims)

    final_split_index = getattr(fedext_model, "layer_split_index", None)
    final_local_layers = [name for name, _ in (fedext_model.local_layers or [])]

    local_modules = []
    applied_projection_shape = None
    if projection_added:
        target_shape = projection_target_shape or original_feature_shape
        if target_shape is None:
            raise ValueError("Cannot determine projection target shape for spatial tail")
        target_dim = int(math.prod(target_shape))
        projection_layers = []
        if not dataset_is_flat:
            projection_layers.append(nn.Flatten(start_dim=1))
        projection_layers.append(nn.Linear(input_feature_dim, target_dim))
        if len(target_shape) > 1:
            projection_layers.append(nn.Unflatten(1, target_shape))
        projection = nn.Sequential(*projection_layers)
        local_modules.append(projection)
        applied_projection_shape = target_shape

    flatten_inserted = False
    for layer_name, layer in fedext_model.local_layers:
        copied_layer = copy.deepcopy(layer)
        local_modules.append(copied_layer)

        if (
            fedext_model.model_type == "resnet"
            and "avgpool" in layer_name
        ):
            local_modules.append(nn.Flatten(start_dim=1))
            flatten_inserted = True

    if len(local_modules) == 0:
        raise ValueError("No local layers found after splitting; check encoder_ratio")

    classifier = nn.Sequential(*local_modules) if len(local_modules) > 1 else local_modules[0]
    classifier = classifier.to(device)

    # Adjust linear head input features if downstream layers changed spatial dimensions.
    final_linear = None
    if isinstance(classifier, nn.Sequential):
        maybe_linear = list(classifier.children())[-1]
        if isinstance(maybe_linear, nn.Linear):
            final_linear = maybe_linear
    elif isinstance(classifier, nn.Linear):
        final_linear = classifier

    head_dim_adjusted_from = None
    if final_linear is not None:
        sample_shape = None
        if not dataset_is_flat and tail_requires_spatial:
            sample_shape = original_feature_shape or input_shape
        else:
            sample_shape = (input_feature_dim,)

        if sample_shape and all(dim is not None for dim in sample_shape):
            dummy = torch.zeros((1, *sample_shape), device=device)
            training_state = classifier.training
            classifier.eval()
            with torch.no_grad():
                if isinstance(classifier, nn.Sequential) and len(classifier) > 1:
                    features_before_head = classifier[:-1](dummy)
                else:
                    features_before_head = dummy
            classifier.train(training_state)

            if isinstance(features_before_head, torch.Tensor):
                flattened = features_before_head.reshape(features_before_head.size(0), -1)
                actual_dim = flattened.size(1)
                if actual_dim != final_linear.in_features:
                    head_dim_adjusted_from = final_linear.in_features
                    new_linear = nn.Linear(actual_dim, final_linear.out_features, bias=final_linear.bias is not None).to(device)
                    if final_linear.bias is not None:
                        with torch.no_grad():
                            new_linear.bias.copy_(final_linear.bias.detach())
                    if isinstance(classifier, nn.Sequential):
                        classifier[-1] = new_linear
                    else:
                        classifier = new_linear
                    final_linear = new_linear

    current_head_dim = None
    if final_linear is not None:
        current_head_dim = final_linear.in_features
    else:
        current_head_dim = model_feature_dim

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
        "model_feature_dim": current_head_dim,
        "input_feature_dim": input_feature_dim,
        "requires_flatten_input": False,
        "requires_reshape_input": False,
        "projection_shape": tuple(applied_projection_shape) if applied_projection_shape else None,
    }

    if head_dim_adjusted_from is not None:
        info["head_in_features_adjusted_from"] = head_dim_adjusted_from
        info["head_in_features_adjusted_to"] = current_head_dim
    info["flatten_inserted_after_avgpool"] = flatten_inserted
    if projection_reason is not None:
        info["projection_reason"] = projection_reason

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
        reason = classifier_info.get("projection_reason")
        if reason == "channel_mismatch":
            reason_note = "（因通道不匹配自动插入）"
        elif reason == "flat_dataset":
            reason_note = "（输入已展平，自动恢复空间形状）"
        else:
            reason_note = ""
        print(f"使用线性映射适配尾部卷积层{proj_shape_str}{reason_note}。")
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
    print(f"尾部期望向量维度: {classifier_info['model_feature_dim']}")
    if classifier_info.get("requires_flatten_input"):
        print("输入特征将自动展平成二维后再送入尾部模型。")
    if classifier_info.get("requires_reshape_input") and classifier_info.get("original_feature_shape"):
        print(f"输入特征将在训练/评估前重塑为 {classifier_info['original_feature_shape']}。")
    if classifier_info.get("head_in_features_adjusted_from") is not None:
        old_dim = classifier_info["head_in_features_adjusted_from"]
        new_dim = classifier_info.get("head_in_features_adjusted_to", old_dim)
        print(f"检测到分类头输入维度不匹配，已自动从 {old_dim} 调整为 {new_dim}。")

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

        test_acc = evaluate(classifier, test_features, test_labels, device, classifier_info)

        if args.use_wandb:
            wandb.log(
                {
                    "Server/step": epoch,
                    "Server/classifier_epoch": epoch,
                    "Server/classifier_train_loss": avg_loss,
                    "Server/classifier_train_accuracy": train_acc,
                    "Server/global_model_test_accuracy": test_acc,
                    "Server/classifier_learning_rate": args.learning_rate,
                }
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, "
            f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}"
        )

    final_test_acc = evaluate(classifier, test_features, test_labels, device, classifier_info)

    print(f"Final test accuracy: {final_test_acc:.4f} on {test_labels.size(0)} samples")

    if args.use_wandb:
        wandb.log(
            {
                "Server/step": args.epochs,
                "Server/global_model_test_accuracy": final_test_acc,
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
