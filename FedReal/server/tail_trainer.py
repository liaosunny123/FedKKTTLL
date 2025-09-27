import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


class LinearTail(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
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
    config: TailTrainConfig,
    epoch_log_hook: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, object]:
    if train_features.numel() == 0:
        raise ValueError("Empty training features received by tail trainer")

    device = torch.device(config.device)

    train_x = train_features.view(train_features.size(0), -1).to(torch.float32)
    train_y = train_labels.to(torch.long)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    test_loader: Optional[DataLoader] = None
    if test_features is not None and test_labels is not None and test_features.numel() > 0:
        test_x = test_features.view(test_features.size(0), -1).to(torch.float32)
        test_y = test_labels.to(torch.long)
        test_dataset = TensorDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = LinearTail(config.feature_dim, config.num_classes).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    logger.info(
        "Tail training — samples=%s, feature_dim=%s, epochs=%s, batch_size=%s",
        train_x.size(0),
        config.feature_dim,
        config.epochs,
        config.batch_size,
    )

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        epoch_total = 0
        correct = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_y.size(0)
            epoch_total += batch_y.size(0)
            predicted = torch.argmax(logits, dim=1)
            correct += (predicted == batch_y).sum().item()

        if epoch_total > 0:
            epoch_loss_avg = epoch_loss / epoch_total
            epoch_acc = correct / epoch_total if epoch_total > 0 else 0.0
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
                test_loss_epoch, test_acc_epoch = _evaluate(model, test_loader, device)
                epoch_metrics["test_loss"] = test_loss_epoch
                epoch_metrics["test_acc"] = test_acc_epoch
                log_parts.append(f"test_loss={test_loss_epoch:.4f}")
                log_parts.append(f"test_acc={test_acc_epoch:.4f}")
                model.train()

            logger.info(" — ".join(log_parts))

            if epoch_log_hook is not None:
                try:
                    epoch_log_hook(epoch + 1, epoch_metrics)
                except Exception as exc:  # pragma: no cover
                    logger.warning("Tail epoch logging hook failed: %s", exc)

    model.eval()
    train_loss, train_acc = _evaluate(model, loader, device)

    eval_results: Optional[Tuple[float, float]] = None
    if test_loader is not None:
        eval_results = _evaluate(model, test_loader, device)

    metrics = {
        "train_loss": train_loss,
        "train_acc": train_acc,
    }
    if eval_results is not None:
        metrics["test_loss"], metrics["test_acc"] = eval_results

    return {
        "state_dict": model.state_dict(),
        "metrics": metrics,
    }
