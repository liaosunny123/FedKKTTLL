# server/server_main.py
import argparse
import concurrent.futures
import logging
import os
import time
import threading
from collections import defaultdict
from typing import Optional

import grpc
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from proto import fed_pb2, fed_pb2_grpc  # 由 protoc 生成
from common.config import FedConfig
from common.utils import set_seed, setup_logger, fmt_bytes
from server.aggregator import Aggregator

# from common.dataset.data_loader import make_global_loaders
# from common.dataset.data_transform import get_transform
from common.data_utils.data_loader import server_data_loader

class FederatedService(fed_pb2_grpc.FederatedServiceServicer):
    def __init__(self, cfg: FedConfig, public_test_loader, device: str = "cpu", run_dir: Optional[str] = None):
        self.cfg = cfg
        self.aggregator = Aggregator(
            cfg,
            public_test_loader=public_test_loader,
            device=device,
            run_dir=run_dir,
        )
        self.logger = logging.getLogger("Server")
        self.start_time = None
        self.end_time = None


        self._byte_lock = threading.Lock()

        self._stage_to_proto = {
            "training": fed_pb2.STAGE_TRAINING,
            "feature_upload": fed_pb2.STAGE_FEATURE_UPLOAD,
            "finalize": fed_pb2.STAGE_FINALIZE,
        }

        # 计算传输成本
        # 总量
        self.bytes_down_global_total = 0          # 下发全局模型总字节（Server→Clients）
        self.bytes_up_local_total = 0             # 回传本地模型总字节（Clients→Server）
        # 分客户端
        self.bytes_down_global_by_client = defaultdict(int)
        self.bytes_up_local_by_client = defaultdict(int)
        self.bytes_up_features_total = 0
        self.bytes_up_features_by_client = defaultdict(int)

    def _cfg_to_proto(self) -> fed_pb2.TrainingConfig:
        return fed_pb2.TrainingConfig(
            num_clients=self.cfg.num_clients,
            total_rounds=self.cfg.total_rounds,
            local_epochs=self.cfg.local_epochs,
            batch_size=self.cfg.batch_size,
            lr=self.cfg.lr,
            momentum=self.cfg.momentum,
            sample_fraction=self.cfg.sample_fraction,
            model_name=self.cfg.model_name,
        )

    # ---- RPC: 客户端注册 ----
    def RegisterClient(self, request, context):
        client_id, client_index, group_index = self.aggregator.register(request.client_name)
        self.logger.info(
            f"Registered {client_id} (index={client_index}, group={group_index})"
        )
        return fed_pb2.RegisterReply(
            client_id=client_id,
            client_index=client_index,
            group_index=group_index,
            config=self._cfg_to_proto(),
        )

    # ---- RPC: 下发训练任务（全局模型+配置）----
    def GetTask(self, request, context):
        round_id, participate, global_bytes, stage, feature_cfg = self.aggregator.get_task(request.client_id)

        # 首轮真正开始计时（仅训练阶段）
        if (
            self.start_time is None
            and stage == "training"
            and round_id < self.cfg.total_rounds
            and self.aggregator.expected_updates > 0
        ):
            self.start_time = time.time()
            self.logger.info("Training timer started.")

        if global_bytes and len(global_bytes) > 0:
            n = len(global_bytes)
            self.logger.info(
                "[Send] global_model bytes=%s (%0.2f MB) to %s for round=%s stage=%s",
                n,
                n / 1024 / 1024,
                request.client_id,
                round_id,
                stage,
            )
            with self._byte_lock:
                self.bytes_down_global_total += n
                self.bytes_down_global_by_client[request.client_id] += n

        if stage == "finalize":
            participate = False

        reply = fed_pb2.TaskReply(
            round=round_id,
            participate=participate,
            global_model=global_bytes,
            config=self._cfg_to_proto(),
            stage=self._stage_to_proto.get(stage, fed_pb2.STAGE_TRAINING),
        )

        if feature_cfg:
            reply.feature.batch_size = int(feature_cfg.get("batch_size", 0))
            reply.feature.keep_spatial = bool(feature_cfg.get("keep_spatial", False))
            reply.feature.include_test_split = bool(feature_cfg.get("include_test_split", True))

        return reply

    # ---- RPC: 接收本地更新（模型权重/样本数/指标）----
    def UploadUpdate(self, request, context):
        if request.local_model:
            n = len(request.local_model)
            self.logger.info(f"[Recv] local_model bytes={fmt_bytes(n)} "
                            f"from {request.client_id} for round={request.round}")
            with self._byte_lock:
                self.bytes_up_local_total += n
                self.bytes_up_local_by_client[request.client_id] += n

        train_loss = request.train_loss if request.HasField("train_loss") else None
        train_acc = request.train_acc if request.HasField("train_acc") else None
        test_loss = request.test_loss if request.HasField("test_loss") else None
        test_acc = request.test_acc if request.HasField("test_acc") else None

        ok = self.aggregator.submit_update(
            client_id=request.client_id,
            group_id=request.group_id,
            round_id=request.round,
            local_bytes=request.local_model,
            num_samples=request.num_samples,
            test_acc=test_acc,
            train_acc=train_acc,
            train_loss=train_loss,
            test_loss=test_loss,
        )
        return fed_pb2.UploadReply(accepted=ok, round=self.aggregator.current_round)

    # ---- RPC: 上传客户端特征数据集 ----
    def UploadFeatures(self, request, context):
        payload_size = len(request.payload)
        self.logger.info(
            "[Recv] features bytes=%s (%0.2f MB) from %s",
            payload_size,
            payload_size / 1024 / 1024,
            request.client_id,
        )

        accepted = self.aggregator.submit_features(
            client_id=request.client_id,
            client_index=request.client_index,
            payload_bytes=request.payload,
            chunk_id=getattr(request, "chunk_id", 0),
            total_chunks=getattr(request, "total_chunks", 0),
        )

        if accepted:
            with self._byte_lock:
                self.bytes_up_features_total += payload_size
                self.bytes_up_features_by_client[request.client_id] += payload_size

        return fed_pb2.FeatureUploadReply(accepted=accepted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", type=str, default="0.0.0.0:50051")
    parser.add_argument("--data_root", type=str, default="./dataset")
    parser.add_argument("--dataset_name", type=str, default="Cifar10", choices=["Cifar10", "Cifar100", "NWPU-RESISC45", "DOTA"],
                        help="Dataset name under dataset/, used to build public test loader")
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--num_clients", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--encoder_ratio", type=float, default=1.0)
    parser.add_argument("--algorithm", type=str, default="FedEXT", choices=["FedEXT", "FedAvg"], help="Algorithm name for logging")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--sample_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="resnet18")
    parser.add_argument("--max_message_mb", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers (None = auto)")
    parser.add_argument("--run_dir", type=str, default=None, help="Directory to store aggregated FedEXT artifacts")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--feature_batch_size", type=int, default=128, help="Batch size used when clients generate feature datasets")
    parser.add_argument("--feature_keep_spatial", action="store_true", help="Ask clients to keep spatial feature maps instead of flattening")
    parser.add_argument("--feature_no_test_split", action="store_true", help="Disable uploading client test split features")
    parser.add_argument("--tail_batch_size", type=int, default=64, help="Batch size for server-side tail training")
    parser.add_argument("--tail_epochs", type=int, default=20, help="Epochs for server-side tail training")
    parser.add_argument("--tail_lr", type=float, default=0.01, help="Learning rate for tail classifier")
    parser.add_argument("--tail_momentum", type=float, default=0.9, help="Momentum for tail classifier optimizer")
    parser.add_argument("--tail_weight_decay", type=float, default=1e-4, help="Weight decay for tail classifier optimizer")
    parser.add_argument("--tail_device", type=str, default=None, help="Device for tail classifier training (default: follow --device)")
    parser.add_argument("--tail_model_name", type=str, default=None, help="Model name used for server-side tail reconstruction (default: follow --model_name)")
    args = parser.parse_args()

    set_seed(args.seed)
    # 统一初始化命名 logger
    logger = setup_logger("Server", level=logging.INFO)

    cfg = FedConfig(
        num_clients=args.num_clients,
        total_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        sample_fraction=args.sample_fraction,
        model_name=args.model_name,
        feature_dim=args.feature_dim,
        encoder_ratio=args.encoder_ratio
    )

    server_test_loader = server_data_loader(args.data_root, args.dataset_name, args.batch_size)
        
    num_classes = args.num_classes

    cfg.num_classes = num_classes
    cfg.max_message_mb = args.max_message_mb
    cfg.dataset_name = args.dataset_name
    cfg.algorithm = args.algorithm
    cfg.use_wandb = args.use_wandb
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.wandb_run_name = args.wandb_run_name
    cfg.feature_batch_size = args.feature_batch_size
    cfg.feature_keep_spatial = args.feature_keep_spatial
    cfg.feature_include_test_split = not args.feature_no_test_split
    cfg.tail_batch_size = args.tail_batch_size
    cfg.tail_epochs = args.tail_epochs
    cfg.tail_lr = args.tail_lr
    cfg.tail_momentum = args.tail_momentum
    cfg.tail_weight_decay = args.tail_weight_decay
    cfg.tail_device = args.tail_device
    cfg.tail_model_name = args.tail_model_name

    service = FederatedService(
        cfg,
        public_test_loader=server_test_loader,
        device=args.device,
        run_dir=args.run_dir,
    )

    max_len = cfg.max_message_mb * 1024 * 1024
    server = grpc.server(
        concurrent.futures.ThreadPoolExecutor(max_workers=32),
        options=[
            ("grpc.max_send_message_length", max_len),
            ("grpc.max_receive_message_length", max_len),
        ],
    )

    # 兼容不同 grpcio-tools 生成的函数名
    add_fn = getattr(fed_pb2_grpc, "add_FederatedServiceServicer_to_server", None)
    if add_fn is None:
        add_fn = fed_pb2_grpc.add_FederatedServiceServicerToServer
    add_fn(service, server)

    server.add_insecure_port(args.bind)
    server.start()
    logger.info(f"Current setting:")
    logger.info(f" - dataset : {args.dataset_name}")
    logger.info(f" - clients number : {args.num_clients}")
    logger.info(f" - local epochs : {args.local_epochs}")
    logger.info(f"Listening on {args.bind}; device={args.device}")
    if args.run_dir:
        logger.info(f"Artifacts will be stored in: {args.run_dir}")

    # 阶段状态指示
    printed_done = False
    training_logged = False
    feature_wait_logged = False
    tail_training_logged = False
    should_exit = False
    try:
        while True:
            time.sleep(1)
            phase = getattr(service.aggregator, "phase", "training")

            if phase == "training":
                if (
                    not training_logged
                    and service.aggregator.current_round >= cfg.total_rounds
                    and service.aggregator.expected_updates == 0
                    and not service.aggregator.selected_this_round
                ):
                    if service.start_time is not None:
                        service.end_time = time.time()
                        elapsed = service.end_time - service.start_time
                        logger.info(
                            f"Federated rounds complete. Training time: {elapsed:.2f}s ({elapsed/60:.2f} min)."
                        )
                    else:
                        logger.info("Federated rounds complete. Awaiting feature uploads...")
                    training_logged = True
                continue

            if phase == "feature_collection":
                if not feature_wait_logged:
                    logger.info("Awaiting client feature uploads for tail training...")
                    feature_wait_logged = True
                continue

            if phase == "server_training":
                if not tail_training_logged:
                    logger.info("All features received. Training server-side tail classifier...")
                    tail_training_logged = True
                continue

            if phase == "finalize":
                if not printed_done:
                    with service._byte_lock:
                        down = service.bytes_down_global_total
                        up_local = service.bytes_up_local_total
                        up_features = service.bytes_up_features_total

                    logger.info(
                        "[Traffic Summary] "
                        f"down_global={fmt_bytes(down)}, "
                        f"up_local={fmt_bytes(up_local)}, "
                        f"up_features={fmt_bytes(up_features)}, "
                        f"total={fmt_bytes(down + up_local + up_features)}"
                    )
                    for cid, v in service.bytes_down_global_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} down_global={fmt_bytes(v)}")
                    for cid, v in service.bytes_up_local_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} up_local={fmt_bytes(v)}")
                    for cid, v in service.bytes_up_features_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} up_features={fmt_bytes(v)}")

                    logger.info(f"Client average acc: {service.aggregator.client_average_test_acc}")
                    logger.info(f"Server total acc : {service.aggregator.server_eval_acc}")
                    logger.info(f"Server total loss : {service.aggregator.server_eval_loss}")

                    if service.aggregator.tail_metrics:
                        logger.info(f"Tail metrics     : {service.aggregator.tail_metrics}")
                    if getattr(service.aggregator, "tail_classifier_info", None):
                        logger.info(f"Tail classifier info: {service.aggregator.tail_classifier_info}")

                    printed_done = True
                    should_exit = True
                break
    except KeyboardInterrupt:
        pass
    finally:
        # 留出缓冲让仍在进行的 RPC 正常结束，避免线程池在关闭后收到请求
        try:
            server.stop(5).wait()
        except Exception:
            logger.exception("Failed to stop gRPC server gracefully")

    if should_exit:
        return


if __name__ == "__main__":
    main()
