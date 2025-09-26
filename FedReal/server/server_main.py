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

        # 计算传输成本
        # 总量
        self.bytes_down_global_total = 0          # 下发全局模型总字节（Server→Clients）
        self.bytes_up_local_total = 0             # 回传本地模型总字节（Clients→Server）
        # 分客户端
        self.bytes_down_global_by_client = defaultdict(int)
        self.bytes_up_local_by_client = defaultdict(int)

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
        round_id, participate, global_bytes = self.aggregator.get_task(request.client_id)

        # 首轮真正开始计时
        if self.start_time is None and round_id < self.cfg.total_rounds and self.aggregator.expected_updates > 0:
            self.start_time = time.time()
            self.logger.info("Training timer started.")

        # ✅ 只有在确实要下发（global_bytes 非空）时才打印/统计
        if global_bytes and len(global_bytes) > 0:
            n = len(global_bytes)
            self.logger.info(f"[Send] global_model bytes={n} ({n/1024/1024:.2f} MB) "
                            f"to {request.client_id} for round={round_id}")
            with self._byte_lock:
                self.bytes_down_global_total += n
                self.bytes_down_global_by_client[request.client_id] += n

        # 结束保护：不参与
        if round_id >= self.cfg.total_rounds:
            participate = False

        return fed_pb2.TaskReply(
            round=round_id,
            participate=participate,
            global_model=global_bytes,  # 可能是空字节，在aggregator里写死了判断逻辑
            config=self._cfg_to_proto(),
        )

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

    server_test_loader = server_data_loader(args.data_root, args.dataset_name, args.batch_size, )
        
    num_classes = args.num_classes

    cfg.num_classes = num_classes
    cfg.dataset_name = args.dataset_name
    cfg.algorithm = args.algorithm
    cfg.use_wandb = args.use_wandb
    cfg.wandb_project = args.wandb_project
    cfg.wandb_entity = args.wandb_entity
    cfg.wandb_run_name = args.wandb_run_name

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

    # 只打印一次“完成”
    printed_done = False
    try:
        while True:
            time.sleep(1)
            if (service.aggregator.current_round >= cfg.total_rounds
                and service.aggregator.expected_updates == 0
                and not service.aggregator.selected_this_round):
                if not printed_done:
                    if service.start_time is not None:
                        service.end_time = time.time()
                        elapsed = service.end_time - service.start_time
                        logger.info(f"Training completed. Total time: {elapsed:.2f}s ({elapsed/60:.2f} min).")
                    else:
                        logger.info("Training completed.")
                    
                    with service._byte_lock:
                        down = service.bytes_down_global_total
                        up_local = service.bytes_up_local_total
                        
                    logger.info(
                        "[Traffic Summary] "
                        f"down_global={fmt_bytes(down)}, "
                        f"up_local={fmt_bytes(up_local)}, "
                        f"total={fmt_bytes(down + up_local)}"
                    )
                    # 如需查看分客户端明细：
                    for cid, v in service.bytes_down_global_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} down_global={fmt_bytes(v)}")
                    for cid, v in service.bytes_up_local_by_client.items():
                        logger.info(f"[Traffic Detail] {cid} up_local={fmt_bytes(v)}")
                    printed_done = True
                    logger.info(f"Client average acc: {service.aggregator.client_average_test_acc}")
                    logger.info(f"Server total acc : {service.aggregator.server_eval_acc}")
                    logger.info(f"Server total loss : {service.aggregator.server_eval_loss}")
                    print("Press Ctrl-C to exit")
                    break
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(0)


if __name__ == "__main__":
    main()
