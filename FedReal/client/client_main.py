import argparse
import io
import time
from pathlib import Path

import grpc
import torch
import logging
from typing import Optional


from proto import fed_pb2, fed_pb2_grpc
#from common.dataset import build_imagefolder_loaders_for_client
#from common.dataset.data_transform import get_transform
from common.data_utils.data_loader import client_data_loader
from common.model.models_fedext import FedEXTModel
from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.utils import setup_logger, set_seed
from client.trainer import train_local, evaluate


def _move_to_device(x, device):
    if isinstance(x, list):
        return [_move_to_device(item, device) for item in x]
    if isinstance(x, tuple):
        return tuple(_move_to_device(item, device) for item in x)
    if torch.is_tensor(x):
        return x.to(device)
    return x


def _extract_features(model, loader, device, keep_spatial):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = _move_to_device(x, device)
            y = y.to(device)
            feats, _ = model.forward_split(x)
            if not keep_spatial and feats.ndim > 2:
                feats = torch.flatten(feats, 1)
            features.append(feats.detach().cpu().to(torch.float32))
            labels.append(y.detach().cpu().to(torch.long))

    if not features:
        return (
            torch.empty((0, model.feature_dim if hasattr(model, "feature_dim") else 0), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
        )

    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def build_feature_payload(
    model,
    train_loader,
    test_loader,
    device,
    keep_spatial: bool,
    include_test: bool,
    metadata: dict,
):
    train_features, train_labels = _extract_features(model, train_loader, device, keep_spatial)
    test_features = torch.empty(0)
    test_labels = torch.empty(0, dtype=torch.long)

    if include_test and test_loader is not None:
        test_features, test_labels = _extract_features(model, test_loader, device, keep_spatial)

    payload = {
        "train_features": train_features,
        "train_labels": train_labels,
        "metadata": metadata,
    }

    if include_test:
        payload["test_features"] = test_features
        payload["test_labels"] = test_labels

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def main():
    
    start_time = None
    end_time = None

    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:50051")
    parser.add_argument("--client_name", type=str, default="client")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dataset_name", type=str, default="Cifar10", choices=["Cifar10", "Cifar100", "NWPU-RESISC45", "DOTA"],
                        help="Dataset name under data/, e.g., cifar10, nwpu, dota")
    parser.add_argument("--num_classes", type=int)                        
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Dataloader workers (None = auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder_ratio", type=float, default=1.0)
    parser.add_argument("--algorithm", type=str, default="FedEXT")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Optional directory to store the final aggregated model copy")
    parser.add_argument("--max_message_mb", type=int, default=128,
                        help="gRPC message size limit in MB for client<->server communication")
    
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logger(f"Client-{args.client_name}", level=logging.INFO)

    # gRPC 消息大小限制（需与服务端一致，避免特征上传时超限）
    max_len = int(args.max_message_mb) * 1024 * 1024
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", max_len),
            ("grpc.max_receive_message_length", max_len),
        ],
    )
    stub = fed_pb2_grpc.FederatedServiceStub(channel)

    # 注册，获得 client_id / client_index / 配置
    reg = stub.RegisterClient(fed_pb2.RegisterRequest(client_name=args.client_name))
    client_id = reg.client_id
    client_index = reg.client_index
    group_index = reg.group_index
    cfg = reg.config

    logger.info(
        f"[Client {client_id}] index={client_index}; group={group_index}; device={args.device}"
    )

    train_loader, test_loader = client_data_loader(args.data_root, args.dataset_name, client_index, args.batch_size)

    # logger.info(f"[Client {client_id}] group_id={group_id}")

    device = torch.device(args.device)

    """ train_labels = [y for _, y in train_loader.dataset.dataset.samples]  # 原始 dataset 的所有标签
    train_indices = train_loader.dataset.indices                        # 当前 client 的样本索引
    client_labels = [train_labels[i] for i in train_indices]

    label_dist = collections.Counter(client_labels)
    logger.info(f"[Client {client_id}] Data allocation: "
                f"train_size={train_size}, test_size={len(test_loader.dataset)}")
    logger.info(f"[Client {client_id}] Label distribution: {dict(label_dist)}") """

    client_pre_eval_acc = []
    client_pre_eval_loss = []
    client_post_eval_acc = []
    client_post_eval_loss = []

    # 主循环
    last_round = -1
    train_size = len(train_loader.dataset)
    model: Optional[FedEXTModel] = None
    feature_uploaded = False

    while True:
        task = stub.GetTask(fed_pb2.GetTaskRequest(client_id=client_id))
        stage = task.stage if hasattr(task, "stage") else fed_pb2.STAGE_TRAINING

        if stage == fed_pb2.STAGE_TRAINING:
            if start_time is None and task.round < cfg.total_rounds:
                start_time = time.time()
                logger.info("Training timer started.")

            if task.round != last_round:
                logger.info(f"[Client {client_id}] Enter round {task.round}")
                last_round = task.round

            if not task.participate:
                if task.global_model:
                    if model is None:
                        model = FedEXTModel(
                            client_index,
                            cfg.model_name,
                            args.feature_dim,
                            args.num_classes,
                            args.encoder_ratio,
                        ).to(device)
                    state_dict = bytes_to_state_dict(task.global_model)
                    model.load_state_dict(state_dict, strict=False)
                time.sleep(1.0)
                continue

            if model is None:
                model = FedEXTModel(
                    client_index,
                    task.config.model_name,
                    args.feature_dim,
                    args.num_classes,
                    args.encoder_ratio,
                ).to(device)

            if task.global_model:
                state_dict = bytes_to_state_dict(task.global_model)
                model.load_state_dict(state_dict, strict=False)

            pre_test_loss, pre_test_acc = evaluate(model, test_loader, device=device)
            logger.info(
                f"[Client {client_id}][Round {task.round}] (pre-trained) test_acc={pre_test_acc:.4f}"
            )
            client_pre_eval_acc.append(pre_test_acc)
            client_pre_eval_loss.append(pre_test_loss)

            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=task.config.lr,
                momentum=task.config.momentum,
                weight_decay=5e-4,
            )
            train_loss_post, train_acc_post = train_local(
                model,
                train_loader,
                epochs=task.config.local_epochs,
                optimizer=optimizer,
                device=device,
            )
            test_loss, test_acc = evaluate(model, test_loader, device=device)
            logger.info(
                f"[Client {client_id}][Round {task.round}] (post-trained) "
                f"train_acc={train_acc_post:.4f} test_acc={test_acc:.4f}"
            )

            client_post_eval_acc.append(test_acc)
            client_post_eval_loss.append(test_loss)

            local_bytes = state_dict_to_bytes(model.state_dict())
            reply = stub.UploadUpdate(
                fed_pb2.UploadRequest(
                    client_id=client_id,
                    group_id=group_index,
                    round=task.round,
                    local_model=local_bytes,
                    num_samples=train_size,
                    train_loss=pre_test_loss,
                    train_acc=pre_test_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                )
            )

            time.sleep(0.2 if reply.accepted else 1.0)
            continue

        if stage == fed_pb2.STAGE_FEATURE_UPLOAD:
            has_feature_cfg = task.HasField("feature") if hasattr(task, "HasField") else False
            feature_batch_size = (
                task.feature.batch_size if has_feature_cfg and task.feature.batch_size > 0 else args.batch_size
            )
            keep_spatial = bool(task.feature.keep_spatial) if has_feature_cfg else False
            include_test = bool(task.feature.include_test_split) if has_feature_cfg else True

            if model is None:
                model = FedEXTModel(
                    client_index,
                    cfg.model_name,
                    args.feature_dim,
                    args.num_classes,
                    args.encoder_ratio,
                ).to(device)
                if task.global_model:
                    state_dict = bytes_to_state_dict(task.global_model)
                    model.load_state_dict(state_dict, strict=False)

            if task.participate and not feature_uploaded:
                feature_train_loader, feature_test_loader = client_data_loader(
                    args.data_root,
                    args.dataset_name,
                    client_index,
                    feature_batch_size,
                )

                metadata = {
                    "client_index": client_index,
                    "dataset": args.dataset_name,
                    "batch_size": feature_batch_size,
                    "keep_spatial": keep_spatial,
                    "include_test": include_test,
                    "encoder_ratio": args.encoder_ratio,
                }

                payload_bytes = build_feature_payload(
                    model,
                    feature_train_loader,
                    feature_test_loader if include_test else None,
                    device,
                    keep_spatial,
                    include_test,
                    metadata,
                )

                max_bytes = int(args.max_message_mb) * 1024 * 1024
                margin = 512 * 1024  # 512KB 裕量，确保不会触达上限
                chunk_size = max(1, max_bytes - margin)
                total_chunks = max(1, (len(payload_bytes) + chunk_size - 1) // chunk_size)

                acked = False
                for chunk_id in range(total_chunks):
                    start = chunk_id * chunk_size
                    end = min(len(payload_bytes), start + chunk_size)
                    chunk = payload_bytes[start:end]

                    feature_reply = stub.UploadFeatures(
                        fed_pb2.FeatureUploadRequest(
                            client_id=client_id,
                            client_index=client_index,
                            payload=chunk,
                            payload_type="torch",
                            chunk_id=chunk_id,
                            total_chunks=total_chunks,
                        )
                    )

                    if chunk_id == total_chunks - 1 and feature_reply.accepted:
                        acked = True
                        logger.info(
                            f"[Client {client_id}] Uploaded feature dataset "
                            f"({total_chunks} chunks, batch_size={metadata['batch_size']})"
                        )

                    if not feature_reply.accepted:
                        logger.warning(
                            f"[Client {client_id}] Feature upload chunk {chunk_id+1}/{total_chunks} "
                            "rejected; will retry later"
                        )
                        break

                feature_uploaded = acked

            time.sleep(1.0)
            continue

        if stage == fed_pb2.STAGE_FINALIZE:
            logger.info(f"[Client {client_id}] Received finalize signal at round {task.round}.")
            if task.global_model:
                if model is None:
                    model = FedEXTModel(
                        client_index,
                        cfg.model_name,
                        args.feature_dim,
                        args.num_classes,
                        args.encoder_ratio,
                    ).to(device)
                state_dict = bytes_to_state_dict(task.global_model)
                model.load_state_dict(state_dict, strict=False)
                if args.run_dir:
                    run_dir = Path(args.run_dir).expanduser().resolve()
                    run_dir.mkdir(parents=True, exist_ok=True)
                    save_path = run_dir / f"client_{client_index:03d}_final.pt"
                    torch.save(model.state_dict(), save_path)
                    logger.info(f"[Client {client_id}] Saved final model to {save_path}")

            if start_time is not None and end_time is None:
                end_time = time.time()
                elapsed = end_time - start_time
                logger.info(
                    f"[Client {client_id}] Total training time: {elapsed:.2f}s ({elapsed/60:.2f} min)"
                )
            break

        time.sleep(1.0)

    print(f"Client pre acc : {client_pre_eval_acc}")
    print(f"Client post acc : {client_post_eval_acc}")
if __name__ == "__main__":
    main()
