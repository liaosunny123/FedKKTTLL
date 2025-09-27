import copy
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.model.models_fedext import FedEXTModel


logger = logging.getLogger("Server").getChild("TailTrainer")


@dataclass
class TailTrainConfig:
    feature_dim: int
    num_classes: int
    batch_size: int = 64
    epochs: int = 20
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 1e-4
    device: str = "cpu"


class _TailLocalModule(nn.Module):
    def __init__(self, local_layers, model_type: str):
        super().__init__()
        self.layer_names = [name for name, _ in local_layers]
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _, layer in local_layers])
        self.model_type = model_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_names:
            first = self.layer_names[0]
            if ("fc" in first or "head" in first) and x.ndim > 2:
                x = torch.flatten(x, 1)
        for name, layer in zip(self.layer_names, self.layers):
            if "avgpool" in name and self.model_type == "resnet":
                x = layer(x)
                x = torch.flatten(x, 1)
            elif "fc" in name and x.ndim > 2:
                x = layer(torch.flatten(x, 1))
            else:
                x = layer(x)
        return x


def build_full_tail_classifier(
    fed_model: FedEXTModel,
    train_feature_shape: Tuple[int, ...],
    dataset_is_flat: bool,
    num_classes: int,
    original_feature_shape: Optional[Tuple[int, ...]] = None,
    force_linear_projection: bool = False,
) -> Tuple[nn.Module, Dict[str, object]]:
    local_layers = fed_model.local_layers or []
    info: Dict[str, object] = {
        "tail_type": None,
        "layer_names": [name for name, _ in local_layers],
        "model_type": fed_model.model_type,
        "input_shape": list(train_feature_shape),
        "original_feature_shape": list(original_feature_shape) if original_feature_shape else None,
    }

    if not local_layers:
        input_dim = int(train_feature_shape[0]) if len(train_feature_shape) == 1 else fed_model.head.in_features
        classifier = nn.Linear(input_dim, num_classes)
        info["tail_type"] = "linear_only"
        info["flatten_input"] = True
        return classifier, info

    first_local_name = local_layers[0][0]
    expects_linear = ("head" in first_local_name) or ("fc" in first_local_name)
    tail_requires_spatial = not expects_linear

    if tail_requires_spatial and dataset_is_flat:
        if not force_linear_projection:
            logger.warning(
                "Tail expects spatial features but received flattened tensors; falling back to linear classifier."
            )
            input_dim = int(train_feature_shape[0]) if len(train_feature_shape) == 1 else fed_model.head.in_features
            classifier = nn.Linear(input_dim, num_classes)
            info["tail_type"] = "linear_fallback"
            info["flatten_input"] = True
            return classifier, info
        if original_feature_shape:
            info["reshape_dims"] = list(original_feature_shape)
        else:
            raise ValueError("Cannot reconstruct spatial features for tail training; missing original shape.")

    classifier = _TailLocalModule(local_layers, fed_model.model_type)
    info["tail_type"] = "local_layers"
    if expects_linear and len(train_feature_shape) > 1:
        info["flatten_input"] = True
    if tail_requires_spatial and train_feature_shape:
        info["reshape_dims"] = info.get("reshape_dims") or list(train_feature_shape)
    return classifier, info


def prepare_features_for_classifier(
    features: torch.Tensor,
    info: Dict[str, object],
    device: torch.device,
) -> torch.Tensor:
    feats = features.to(torch.float32)
    reshape_dims = info.get("reshape_dims")
    if reshape_dims and feats.ndim == 2:
        feats = feats.view(feats.size(0), *reshape_dims)
    if info.get("flatten_input") and feats.ndim > 2:
        feats = feats.view(feats.size(0), -1)
    return feats.to(device)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classifier_info: Dict[str, object],
) -> Tuple[float, float]:
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch_x, batch_y in loader:
        batch_x = prepare_features_for_classifier(batch_x, classifier_info, device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss_sum += loss.item() * batch_y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == batch_y).sum().item()
        total += batch_y.size(0)
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def train_tail_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: Optional[torch.Tensor],
    test_labels: Optional[torch.Tensor],
    classifier: nn.Module,
    classifier_info: Dict[str, object],
    config: TailTrainConfig,
    epoch_log_hook: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    if train_features.numel() == 0:
        raise ValueError("Empty training features received by tail trainer")

    device = torch.device(config.device)
    classifier = classifier.to(device)

    train_dataset = TensorDataset(train_features, train_labels.to(torch.long))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_loader: Optional[DataLoader] = None
    if test_features is not None and test_labels is not None and test_features.numel() > 0:
        test_dataset = TensorDataset(test_features, test_labels.to(torch.long))
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    logger.info(
        "Tail training — samples=%s, feature_shape=%s, epochs=%s, batch_size=%s",
        train_features.shape[0],
        list(train_features.shape[1:]),
        config.epochs,
        config.batch_size,
    )

    classifier.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_total = 0
        correct = 0
        for batch_x, batch_y in train_loader:
            inputs = prepare_features_for_classifier(batch_x, classifier_info, device)
            labels = batch_y.to(device)

            optimizer.zero_grad()
            logits = classifier(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            epoch_total += labels.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == labels).sum().item()

        if epoch_total > 0:
            epoch_loss_avg = epoch_loss / epoch_total
            epoch_acc = correct / epoch_total
            epoch_metrics: Dict[str, float] = {
                "train_loss": epoch_loss_avg,
                "train_acc": epoch_acc,
            }

            log_parts = [
                f"Tail epoch {epoch + 1}/{config.epochs}",
                f"loss={epoch_loss_avg:.4f}",
                f"train_acc={epoch_acc:.4f}",
            ]

            if test_loader is not None:
                classifier.eval()
                test_loss_epoch, test_acc_epoch = _evaluate(classifier, test_loader, device, classifier_info)
                epoch_metrics["test_loss"] = test_loss_epoch
                epoch_metrics["test_acc"] = test_acc_epoch
                log_parts.append(f"test_loss={test_loss_epoch:.4f}")
                log_parts.append(f"test_acc={test_acc_epoch:.4f}")
                classifier.train()

            logger.info(" — ".join(log_parts))

            if epoch_log_hook is not None:
                try:
                    epoch_log_hook(epoch + 1, epoch_metrics)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Tail epoch logging hook failed: %s", exc)

    classifier.eval()
    train_loss, train_acc = _evaluate(classifier, train_loader, device, classifier_info)

    eval_results: Optional[Tuple[float, float]] = None
    if test_loader is not None:
        eval_results = _evaluate(classifier, test_loader, device, classifier_info)

    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
    }
    if eval_results is not None:
        metrics["test_loss"], metrics["test_acc"] = eval_results

    return {
        "state_dict": classifier.cpu().state_dict(),
        "metrics": metrics,
        "classifier_info": classifier_info,
    }
