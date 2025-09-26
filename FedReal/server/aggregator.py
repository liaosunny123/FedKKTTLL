import json
import logging
import math
import threading
import time
from collections import defaultdict, OrderedDict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Set, Tuple

import torch

from common.model.models_fedext import FedEXTModel
from common.serialization import bytes_to_state_dict, state_dict_to_bytes
from common.utils import select_clients

from .eval import evaluate


logger = logging.getLogger("Server").getChild("Aggregator")


class Aggregator:
    def __init__(self, config, public_test_loader=None, device: str = "cpu", run_dir: Optional[str] = None):
        self.cfg = config
        self.device = torch.device(device)
        self.public_test_loader = public_test_loader

        # —— FedEXT specific attributes ——
        self.encoder_ratio = float(getattr(self.cfg, "encoder_ratio", 1.0))
        default_group_count = min(5, getattr(self.cfg, "num_clients", 1))
        self.group_count = max(1, int(getattr(self.cfg, "group_count", default_group_count)))

        # —— Runtime bookkeeping ——
        self.current_round = 0
        self.registered: List[str] = []
        self.client_index: Dict[str, int] = {}
        self.client_groups: Dict[str, int] = {}

        self.selected_this_round: List[str] = []
        self.expected_updates = 0
        self.received_updates: Dict[str, Dict[str, object]] = {}
        self.completed_this_round: Set[str] = set()
        self.require_full = True
        self.lock = threading.Lock()

        # —— Metrics ——
        self.client_average_test_acc: List[float] = []
        self.client_average_pre_test_acc: List[float] = []
        self.server_eval_acc: List[float] = []
        self.server_eval_loss: List[float] = []
        self.round_client_metrics: Dict[int, List[Dict[str, Optional[float]]]] = defaultdict(list)

        # —— Model buffers ——
        self._template_model = self._new_model_instance()
        self.layer_split_index = getattr(self._template_model, "layer_split_index", 0)

        self.model = self._new_model_instance()
        self.model.load_state_dict(self._template_model.state_dict())
        self.model.to(self.device)
        self.model.eval()

        self.global_bytes = state_dict_to_bytes(self._cpu_state_dict(self.model.state_dict()))
        self.group_model_states: Dict[int, OrderedDict] = {}
        self.group_model_bytes: Dict[int, bytes] = {}
        self._initialize_group_buffers()

        # —— Persistence ——
        self.run_dir: Optional[Path] = Path(run_dir).expanduser().resolve() if run_dir else None
        if self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "groups").mkdir(exist_ok=True)
            (self.run_dir / "global").mkdir(exist_ok=True)

        # —— WandB ——
        self.use_wandb = bool(getattr(self.cfg, "use_wandb", False))
        self._wandb = None
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb  # type: ignore
            except ImportError as exc:
                raise RuntimeError("use_wandb=True 但未安装 wandb 库") from exc

            run_name = getattr(self.cfg, "wandb_run_name", None) or f"fedreal-server-{int(time.time())}"
            project = getattr(self.cfg, "wandb_project", None) or "fedreal"
            entity = getattr(self.cfg, "wandb_entity", None) or None

            config_payload = {
                "dataset": getattr(self.cfg, "dataset_name", None),
                "num_clients": self.cfg.num_clients,
                "rounds": self.cfg.total_rounds,
                "local_epochs": self.cfg.local_epochs,
                "batch_size": self.cfg.batch_size,
                "lr": self.cfg.lr,
                "momentum": self.cfg.momentum,
                "sample_fraction": self.cfg.sample_fraction,
                "model_name": self.cfg.model_name,
                "feature_dim": self.cfg.feature_dim,
                "encoder_ratio": self.cfg.encoder_ratio,
                "algorithm": getattr(self.cfg, "algorithm", "unknown"),
            }

            self._wandb = wandb
            self.wandb_run = wandb.init(
                project=project,
                entity=entity if entity else None,
                name=run_name,
                config=config_payload,
                reinit=True,
            )
            wandb.define_metric("Global/step")
            wandb.define_metric("Client/*", step_metric="Global/step")

    # ------------------------------------------------------------------
    # Helper construction utilities
    # ------------------------------------------------------------------
    def _new_model_instance(self) -> FedEXTModel:
        model = FedEXTModel(
            cid=-1,
            model_name=self.cfg.model_name,
            feature_dim=self.cfg.feature_dim,
            num_classes=self.cfg.num_classes,
            encoder_ratio=self.encoder_ratio,
        ).to(self.device)
        # Ensure split is applied in case constructor was bypassed
        if hasattr(model, "set_layer_split") and hasattr(model, "layer_split_index"):
            model.set_layer_split(model.layer_split_index)
        return model

    def _initialize_group_buffers(self):
        base_state = self._cpu_state_dict(self.model.state_dict())
        for gid in range(self.group_count):
            state_copy = OrderedDict((k, v.clone()) for k, v in base_state.items())
            self.group_model_states[gid] = state_copy
            self.group_model_bytes[gid] = state_dict_to_bytes(state_copy)

    @staticmethod
    def _cpu_state_dict(state_dict: OrderedDict) -> OrderedDict:
        return OrderedDict((k, v.detach().cpu()) for k, v in state_dict.items())

    @staticmethod
    def _cpu_params(params: Optional[OrderedDict]) -> Optional[OrderedDict]:
        if params is None:
            return None
        return OrderedDict((k, v.detach().cpu()) for k, v in params.items())

    def _assign_group(self, index: int) -> int:
        if self.group_count <= 0:
            return 0
        return index % self.group_count

    def _model_bytes_for_group(self, group_id: int) -> bytes:
        if group_id in self.group_model_bytes:
            return self.group_model_bytes[group_id]
        if self.group_model_bytes:
            # fallback to the first available group model
            return next(iter(self.group_model_bytes.values()))
        return self.global_bytes

    # ------------------------------------------------------------------
    # Federated protocol plumbing
    # ------------------------------------------------------------------
    def register(self, client_name: str) -> Tuple[str, int, int]:
        with self.lock:
            client_id = f"C{len(self.registered):03d}"
            client_idx = len(self.registered)
            self.registered.append(client_id)
            self.client_index[client_id] = client_idx
            group_id = self._assign_group(client_idx)
            self.client_groups[client_id] = group_id
            logger.info(
                f"Registered {client_id} (index={client_idx}) assigned to group {group_id}"
            )
            return client_id, client_idx, group_id

    def _ensure_sampling(self):
        with self.lock:
            if self.current_round >= self.cfg.total_rounds:
                self.selected_this_round = []
                self.expected_updates = 0
                self.received_updates.clear()
                return

            if self.expected_updates > 0 or self.selected_this_round:
                return

            if self.require_full and len(self.registered) < self.cfg.num_clients:
                logger.warning(
                    "Round %s not started: registered=%s < required=%s",
                    self.current_round,
                    len(self.registered),
                    self.cfg.num_clients,
                )
                self.selected_this_round = []
                self.expected_updates = 0
                self.received_updates.clear()
                return

            k_target = max(1, math.ceil(self.cfg.sample_fraction * self.cfg.num_clients))
            k = min(k_target, len(self.registered))
            if self.require_full and k < k_target:
                logger.warning(
                    "Round %s not started: sampled=%s < target=%s",
                    self.current_round,
                    k,
                    k_target,
                )
                self.selected_this_round = []
                self.expected_updates = 0
                self.received_updates.clear()
                return

            self.selected_this_round = select_clients(
                self.registered, k, self.current_round, seed=self.cfg.seed
            )
            self.expected_updates = len(self.selected_this_round)
            self.received_updates.clear()

            if self.selected_this_round:
                logger.info(
                    "Round %s sampling -> %s (expected_updates=%s)",
                    self.current_round,
                    self.selected_this_round,
                    self.expected_updates,
                )
            else:
                logger.warning("Round %s not started (waiting)", self.current_round)

    def get_task(self, client_id: str):
        self._ensure_sampling()
        with self.lock:
            group_id = self.client_groups.get(client_id, 0)

            if self.current_round >= self.cfg.total_rounds:
                model_bytes = self._model_bytes_for_group(group_id)
                return self.current_round, False, model_bytes

            participate = client_id in self.selected_this_round and self.expected_updates > 0

            if client_id in self.completed_this_round:
                participate = False

            model_bytes = self._model_bytes_for_group(group_id) if participate else b""
            return self.current_round, participate, model_bytes

    def submit_update(
        self,
        client_id: str,
        group_id: int,
        round_id: int,
        local_bytes: bytes,
        num_samples: int,
        test_acc: Optional[float] = None,
        train_acc: Optional[float] = None,
        train_loss: Optional[float] = None,
        test_loss: Optional[float] = None,
    ):
        with self.lock:
            logger.info(
                "recv from %s (group=%s) for round=%s (curr=%s) %s/%s",
                client_id,
                group_id,
                round_id,
                self.current_round,
                len(self.received_updates) + 1,
                self.expected_updates,
            )

            if round_id != self.current_round:
                logger.warning(
                    "drop update from %s: stale round=%s (curr=%s)",
                    client_id,
                    round_id,
                    self.current_round,
                )
                return False

            if client_id in self.received_updates:
                logger.debug("ignore duplicate update from %s", client_id)
                return True

            state_dict = bytes_to_state_dict(local_bytes) if local_bytes else OrderedDict()

            if client_id in self.client_groups and group_id != self.client_groups[client_id]:
                logger.warning(
                    "Client %s reported mismatched group %s (expected %s)",
                    client_id,
                    group_id,
                    self.client_groups[client_id],
                )

            self.received_updates[client_id] = {
                "state_dict": state_dict,
                "num_samples": int(num_samples),
                "group_id": int(group_id),
            }
            self.completed_this_round.add(client_id)
            metric_record = {
                "client_id": client_id,
                "group_id": group_id,
                "num_samples": num_samples,
                "pre_test_acc": train_acc,
                "pre_test_loss": train_loss,
                "post_test_acc": test_acc,
                "post_test_loss": test_loss,
            }
            self.round_client_metrics[round_id].append(metric_record)

            if self.use_wandb and self._wandb is not None:
                client_tag = f"Client_{client_id}"
                log_payload = {"Global/step": round_id}
                if train_acc is not None:
                    log_payload[f"{client_tag}/pre_test_acc"] = train_acc
                if train_loss is not None:
                    log_payload[f"{client_tag}/pre_test_loss"] = train_loss
                if test_acc is not None:
                    log_payload[f"{client_tag}/post_test_acc"] = test_acc
                if test_loss is not None:
                    log_payload[f"{client_tag}/post_test_loss"] = test_loss
                if num_samples:
                    log_payload[f"{client_tag}/num_samples"] = num_samples
                self._wandb.log(log_payload, step=round_id)

            if len(self.received_updates) >= self.expected_updates:
                self._aggregate_and_advance()
            return True

    # ------------------------------------------------------------------
    # Aggregation logic
    # ------------------------------------------------------------------
    def _aggregate_and_advance(self):
        if not self.received_updates:
            return

        total_samples = sum(p["num_samples"] for p in self.received_updates.values())
        total_samples = float(total_samples) if total_samples > 0 else 0.0

        round_metrics = self.round_client_metrics.pop(self.current_round, [])
        pre_acc_values = [m["pre_test_acc"] for m in round_metrics if m.get("pre_test_acc") is not None]
        post_acc_values = [m["post_test_acc"] for m in round_metrics if m.get("post_test_acc") is not None]
        pre_loss_values = [m["pre_test_loss"] for m in round_metrics if m.get("pre_test_loss") is not None]
        post_loss_values = [m["post_test_loss"] for m in round_metrics if m.get("post_test_loss") is not None]

        global_params_list: List[Tuple[OrderedDict, float]] = []
        local_params_by_group: Dict[int, List[OrderedDict]] = defaultdict(list)
        weights_by_group: Dict[int, List[float]] = defaultdict(list)

        for client_id, payload in self.received_updates.items():
            state_dict = payload.get("state_dict", OrderedDict())
            group_id = payload.get("group_id", 0)
            num_samples = float(payload.get("num_samples", 0))

            model = self._new_model_instance()
            if state_dict:
                model.load_state_dict(state_dict, strict=False)

            if total_samples > 0:
                weight = num_samples / total_samples
            else:
                weight = 1.0 / max(1, len(self.received_updates))

            global_params = model.get_global_params()
            if global_params:
                global_params_list.append((global_params, weight))

            local_params = model.get_local_params()
            if local_params:
                local_params_by_group[group_id].append(local_params)
                weights_by_group[group_id].append(num_samples)

        aggregated_global = self._fedavg_aggregation(global_params_list)
        aggregated_local_by_group: Dict[int, OrderedDict] = {}

        for group_id, params_list in local_params_by_group.items():
            group_weights = weights_by_group[group_id]
            total_group = sum(group_weights)
            if total_group <= 0:
                normalized = [1.0 / len(params_list)] * len(params_list)
            else:
                normalized = [w / total_group for w in group_weights]
            weighted_params = list(zip(params_list, normalized))
            aggregated_local_by_group[group_id] = self._fedavg_aggregation(weighted_params)

        agg_global_cpu = self._cpu_params(aggregated_global)
        agg_local_cpu = {gid: self._cpu_params(params) for gid, params in aggregated_local_by_group.items()}

        self._update_group_models(agg_global_cpu, agg_local_cpu)

        # Evaluate on server public test set using group 0 (or first available) model
        eval_group = 0 if 0 in self.group_model_states else next(iter(self.group_model_states.keys()))
        eval_state = self.group_model_states[eval_group]
        self.model.load_state_dict(eval_state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        global_eval_loss = None
        global_eval_acc = None
        if self.public_test_loader is not None:
            loss, acc = evaluate(self.model, self.public_test_loader, device=self.device)
            logger.info("[Round %s] Global Eval — loss=%.4f, acc=%.4f", self.current_round, loss, acc)
            self.server_eval_acc.append(acc)
            self.server_eval_loss.append(loss)
            global_eval_loss = loss
            global_eval_acc = acc
        else:
            logger.debug("[Round %s] Global Eval — skipped (no public_test_loader)", self.current_round)

        avg_pre = mean(pre_acc_values) if pre_acc_values else 0.0
        avg_post = mean(post_acc_values) if post_acc_values else 0.0
        self.client_average_pre_test_acc.append(avg_pre)
        self.client_average_test_acc.append(avg_post)

        if self.use_wandb and self._wandb is not None:
            log_payload = {"Global/step": self.current_round}
            if global_eval_acc is not None:
                log_payload["Global/test_acc"] = global_eval_acc
            if global_eval_loss is not None:
                log_payload["Global/test_loss"] = global_eval_loss
            if post_acc_values:
                log_payload["Clients/avg_post_test_acc"] = avg_post
            if pre_acc_values:
                log_payload["Clients/avg_pre_test_acc"] = avg_pre
            if post_loss_values:
                log_payload["Clients/avg_post_test_loss"] = mean(post_loss_values)
            if pre_loss_values:
                log_payload["Clients/avg_pre_test_loss"] = mean(pre_loss_values)
            self._wandb.log(log_payload, step=self.current_round)

        round_finished = self.current_round + 1
        if round_finished == self.cfg.total_rounds:
            self._persist_final_artifacts(round_finished, agg_global_cpu, agg_local_cpu)

        self.current_round += 1
        self.selected_this_round = []
        self.expected_updates = 0
        self.received_updates.clear()
        self.completed_this_round.clear()

    @staticmethod
    def _fedavg_aggregation(weighted_params_list: List[Tuple[OrderedDict, float]]) -> Optional[OrderedDict]:
        if not weighted_params_list:
            return None

        first_params, _ = weighted_params_list[0]
        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()

        for key, tensor in first_params.items():
            if tensor.dtype.is_floating_point:
                acc = torch.zeros_like(tensor, dtype=torch.float32)
                for params, weight in weighted_params_list:
                    if key not in params:
                        continue
                    acc += params[key].to(torch.float32) * weight
                aggregated[key] = acc.to(tensor.dtype)
            else:
                # For non-floating buffers (e.g., num_batches_tracked) keep the first client's value.
                aggregated[key] = tensor.clone()

        return aggregated

    def _update_group_models(
        self,
        aggregated_global: Optional[OrderedDict],
        aggregated_local_by_group: Dict[int, Optional[OrderedDict]],
    ):
        for gid in range(self.group_count):
            prev_state = self.group_model_states.get(gid)
            model = self._new_model_instance()

            if prev_state:
                model.load_state_dict(prev_state, strict=False)

            if aggregated_global:
                model.set_global_params(aggregated_global)

            if gid in aggregated_local_by_group and aggregated_local_by_group[gid]:
                model.set_local_params(aggregated_local_by_group[gid])

            updated_state = self._cpu_state_dict(model.state_dict())
            self.group_model_states[gid] = updated_state
            self.group_model_bytes[gid] = state_dict_to_bytes(updated_state)

        # Update global broadcast model (default to group 0)
        primary_group = 0 if 0 in self.group_model_bytes else next(iter(self.group_model_bytes.keys()))
        self.global_bytes = self.group_model_bytes[primary_group]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _persist_final_artifacts(
        self,
        round_number: int,
        aggregated_global: Optional[OrderedDict],
        aggregated_local_by_group: Dict[int, Optional[OrderedDict]],
    ):
        if not self.run_dir:
            if self.use_wandb and self._wandb is not None and self.wandb_run is not None:
                summary_payload = {
                    "final_round": round_number,
                    "final_global_test_acc": self.server_eval_acc[-1] if self.server_eval_acc else None,
                    "final_global_test_loss": self.server_eval_loss[-1] if self.server_eval_loss else None,
                    "best_global_test_acc": max(self.server_eval_acc) if self.server_eval_acc else None,
                    "best_client_post_test_acc": max(self.client_average_test_acc) if self.client_average_test_acc else None,
                    "final_client_post_test_acc": self.client_average_test_acc[-1] if self.client_average_test_acc else None,
                    "final_client_pre_test_acc": self.client_average_pre_test_acc[-1] if self.client_average_pre_test_acc else None,
                }
                for key, value in summary_payload.items():
                    if value is not None:
                        self.wandb_run.summary[key] = value

                self._wandb.log({"Global/step": round_number, "Global/final_round": round_number}, step=round_number)
                self._wandb.finish()
                self.wandb_run = None
                self._wandb = None
            return

        logger.info("Persisting final FedEXT artifacts to %s", self.run_dir)

        # Save per-client full models (compatible with system/generate_datasets.py)
        for client_id, index in self.client_index.items():
            group_id = self.client_groups.get(client_id, 0)
            state = self.group_model_states.get(group_id)
            if state:
                payload = {
                    "state_dict": state,
                    "model_name": self.cfg.model_name,
                    "feature_dim": self.cfg.feature_dim,
                    "num_classes": self.cfg.num_classes,
                    "encoder_ratio": self.encoder_ratio,
                }
                torch.save(payload, self.run_dir / f"Client_{index}_model.pt")

        # Save per-group models
        groups_dir = self.run_dir / "groups"
        groups_dir.mkdir(exist_ok=True)
        for gid, state in self.group_model_states.items():
            torch.save(state, groups_dir / f"group_{gid}_model.pt")

        # Save global/local parameter subsets for debugging
        global_dir = self.run_dir / "global"
        global_dir.mkdir(exist_ok=True)
        if aggregated_global:
            torch.save(aggregated_global, global_dir / "global_params.pt")
        else:
            torch.save({}, global_dir / "global_params.pt")

        local_dir = self.run_dir / "local"
        local_dir.mkdir(exist_ok=True)
        for gid, params in aggregated_local_by_group.items():
            if params:
                torch.save(params, local_dir / f"group_{gid}_local_params.pt")

        metadata = {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_rounds": int(self.cfg.total_rounds),
            "encoder_ratio": self.encoder_ratio,
            "layer_split_index": int(self.layer_split_index),
            "group_count": int(self.group_count),
            "dataset_name": getattr(self.cfg, "dataset_name", None),
            "num_clients": int(getattr(self.cfg, "num_clients", len(self.registered))),
            "client_groups": {cid: int(self.client_groups.get(cid, 0)) for cid in self.registered},
            "model_name": self.cfg.model_name,
            "feature_dim": self.cfg.feature_dim,
            "num_classes": self.cfg.num_classes,
            "round_finished": round_number,
            "algorithm": getattr(self.cfg, "algorithm", None),
        }

        with open(self.run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if self.use_wandb and self._wandb is not None and self.wandb_run is not None:
            summary_payload = {
                "final_round": round_number,
                "final_global_test_acc": self.server_eval_acc[-1] if self.server_eval_acc else None,
                "final_global_test_loss": self.server_eval_loss[-1] if self.server_eval_loss else None,
                "best_global_test_acc": max(self.server_eval_acc) if self.server_eval_acc else None,
                "best_client_post_test_acc": max(self.client_average_test_acc) if self.client_average_test_acc else None,
                "final_client_post_test_acc": self.client_average_test_acc[-1] if self.client_average_test_acc else None,
                "final_client_pre_test_acc": self.client_average_pre_test_acc[-1] if self.client_average_pre_test_acc else None,
            }
            for key, value in summary_payload.items():
                if value is not None:
                    self.wandb_run.summary[key] = value

            self._wandb.log({"Global/step": round_number, "Global/final_round": round_number}, step=round_number)
            self._wandb.finish()
            self.wandb_run = None
            self._wandb = None
