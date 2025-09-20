import time
import numpy as np
import torch
import torch.nn as nn
import copy
from flcore.servers.serverbase import Server
from flcore.clients.clientfedext import clientFedEXT
from flcore.clients.clientbase import load_item, save_item
from flcore.trainmodel.models_fedext import FedEXTModel
from threading import Thread
import wandb
from collections import defaultdict, OrderedDict


class FedEXT(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedEXT)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedEXT server and clients.")

        self.Budget = []
        self.args = args

        # Get encoder ratio
        self.encoder_ratio = getattr(args, 'encoder_ratio', 0.7)

        # FedEXT specific attributes
        self.groups = {}  # Dictionary to store client groups

        # Initialize client groups (0-19 divided into 5 groups)
        self._initialize_groups()

        # Initialize models and analyze structure
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            # Initialize client models with FedEXT structure
            for client in self.clients:
                model = FedEXTModel(args, client.id).to(self.device)
                save_item(model, client.role, 'model', client.save_folder_name)

                # Set client's group
                if client.id in self.groups:
                    client.set_group(self.groups[client.id])

            # Analyze model structure and determine key grouping
            if len(self.clients) > 0:
                sample_model = FedEXTModel(args, 0).to(self.device)
                self._analyze_model_structure(sample_model)

        print(f"\nClient groups: {self.groups}")
        print(f"Encoder ratio: {self.encoder_ratio}")
        print(f"Aggregation strategy:")
        if self.encoder_ratio >= 0.99:
            print(f"  → Pure FedAvg mode (100% global aggregation)")
        elif self.encoder_ratio <= 0.01:
            print(f"  → Pure Group mode (100% group aggregation)")
        else:
            print(f"  → Hybrid mode:")
            print(f"    - First {self.encoder_ratio*100:.0f}% of layers: global aggregation")
            print(f"    - Last {(1-self.encoder_ratio)*100:.0f}% of layers: group aggregation")

    def _analyze_model_structure(self, sample_model):
        """
        Analyze model structure to determine layer ordering
        PyTorch state_dict() preserves module registration order:
        - Modules are visited in the order they were registered
        - For BaseHeadSplit: base.* keys come before head.* keys
        """
        # Get all parameters in order
        all_state_dict = sample_model.state_dict()
        all_keys_ordered = list(all_state_dict.keys())

        # Get base and head parameters
        base_params = sample_model.get_base_params()
        head_params = sample_model.get_head_params()

        base_keys = set(base_params.keys())
        head_keys = set(head_params.keys())

        # Preserve registration order within base and head
        self.ordered_base_keys = [k for k in all_keys_ordered if k in base_keys]
        self.ordered_head_keys = [k for k in all_keys_ordered if k in head_keys]

        # Combine: base keys first (encoder), then head keys (classifier)
        # This matches the forward pass: base(x) -> head(base_output)
        self.all_keys_ordered = self.ordered_base_keys + self.ordered_head_keys

        print(f"\nModel structure analysis:")
        print(f"  Total parameters: {len(self.all_keys_ordered)} keys")
        print(f"  Base (encoder) parameters: {len(self.ordered_base_keys)} keys")
        print(f"  Head (classifier) parameters: {len(self.ordered_head_keys)} keys")

        # Calculate split point based on encoder_ratio
        total_keys = len(self.all_keys_ordered)
        base_key_count = len(self.ordered_base_keys)
        head_key_count = len(self.ordered_head_keys)

        # More intuitive split calculation
        if self.encoder_ratio >= 0.99:
            # Pure FedAvg: aggregate everything globally
            self.split_point = total_keys
        elif self.encoder_ratio <= 0.01:
            # Pure Group: aggregate everything in groups
            self.split_point = 0
        else:
            # Hybrid: split proportionally
            # encoder_ratio=0.7 means 70% of all parameters are aggregated globally
            self.split_point = int(total_keys * self.encoder_ratio)

            # Ensure we don't split in the middle of base or head unnecessarily
            # If split point is close to base/head boundary, snap to it
            if abs(self.split_point - base_key_count) <= 2:
                self.split_point = base_key_count
                print(f"  Adjusted split to base/head boundary")

        print(f"\n  With encoder_ratio={self.encoder_ratio}:")
        if self.split_point == 0:
            print(f"    - All parameters use group aggregation")
        elif self.split_point == total_keys:
            print(f"    - All parameters use global aggregation")
        else:
            print(f"    - Global aggregation: first {self.split_point} keys")
            print(f"    - Group aggregation: last {total_keys-self.split_point} keys")

            # Show which parts are globally/group aggregated
            global_base = min(self.split_point, base_key_count)
            global_head = max(0, self.split_point - base_key_count)
            group_base = max(0, base_key_count - self.split_point)
            group_head = max(0, total_keys - max(self.split_point, base_key_count))

            if global_base > 0:
                print(f"    - Base: {global_base}/{base_key_count} keys globally aggregated")
            if global_head > 0:
                print(f"    - Head: {global_head}/{head_key_count} keys globally aggregated")
            if group_head > 0:
                print(f"    - Head: {group_head}/{head_key_count} keys group aggregated")
            if group_base > 0:
                print(f"    - Base: {group_base}/{base_key_count} keys group aggregated")

    def _initialize_groups(self):
        """Initialize client groups based on predefined assignment"""
        # Groups 0-4 with 4 clients each
        for i in [0, 5, 10, 15]:
            self.groups[i] = 0
        for i in [1, 6, 11, 16]:
            self.groups[i] = 1
        for i in [2, 7, 12, 17]:
            self.groups[i] = 2
        for i in [3, 8, 13, 18]:
            self.groups[i] = 3
        for i in [4, 9, 14, 19]:
            self.groups[i] = 4

    def train(self):
        # Initial evaluation
        print(f"\n-------------Initial Evaluation-------------")
        print("\nEvaluate initial models performance")
        self.evaluate()

        print(f"\nStarting FedEXT training...")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            print(f"\n-------------Round number: {i}-------------")
            print(f"Clients selected: {[c.id for c in self.selected_clients]}")

            # Train selected clients
            for client in self.selected_clients:
                client.train()

            # Receive models from clients
            self.receive_models()

            # Aggregate models based on encoder_ratio
            self.aggregate_models_by_ratio()

            # Send updated models back to clients
            self.send_models()

            # Evaluation
            if i % self.eval_gap == 0:
                print("\nEvaluate models after aggregation")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, 'Round time cost:', self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print(f"\nBest accuracy:")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round:")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else 0)

        self.save_results()

    def receive_models(self):
        """Receive models from selected clients"""
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_groups = []
        tot_samples = 0

        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_groups.append(self.groups.get(client.id, 0))

            # Load client model
            model = load_item(client.role, 'model', client.save_folder_name)
            self.uploaded_models.append(model.state_dict())

        # Normalize weights
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_models_by_ratio(self):
        """
        Aggregate models based on encoder_ratio
        Split parameters by layer order, not randomly
        """
        print(f"\nAggregating models (encoder_ratio={self.encoder_ratio})...")

        # Special cases
        if self.encoder_ratio >= 0.99:
            # Pure FedAvg: aggregate ALL parameters globally (including head!)
            print("  Mode: Pure FedAvg (100% global aggregation)")
            aggregated_model = self._aggregate_all_globally()
            self._distribute_models(aggregated_model)

        elif self.encoder_ratio <= 0.01:
            # Pure Group: aggregate ALL parameters within groups
            print("  Mode: Pure Group (100% group aggregation)")
            group_models = self._aggregate_all_by_groups()
            self._distribute_group_models(group_models)

        else:
            # Hybrid mode: split by layer order
            print(f"  Mode: Hybrid (split at key {self.split_point})")

            # Split keys based on the pre-calculated split point
            global_keys = self.all_keys_ordered[:self.split_point]
            group_keys = self.all_keys_ordered[self.split_point:]

            print(f"    - Global aggregation: {len(global_keys)} keys")
            print(f"    - Group aggregation: {len(group_keys)} keys")

            # Aggregate global part
            global_params = self._aggregate_parameters_globally(global_keys)

            # Aggregate group part
            group_params = self._aggregate_parameters_by_groups(group_keys)

            # Distribute combined models
            self._distribute_hybrid_models(global_params, global_keys, group_params, group_keys)

    def _aggregate_all_globally(self):
        """Aggregate all parameters globally (FedAvg style)"""
        aggregated = {}
        for key in self.uploaded_models[0].keys():
            aggregated[key] = torch.zeros_like(self.uploaded_models[0][key])

        for w, model_params in zip(self.uploaded_weights, self.uploaded_models):
            for key in aggregated.keys():
                aggregated[key] += model_params[key] * w

        return aggregated

    def _aggregate_all_by_groups(self):
        """Aggregate all parameters within groups"""
        group_models = {}

        # Group models by group ID
        group_params_dict = defaultdict(list)
        group_weights_dict = defaultdict(list)

        for i, (client_id, group_id, model_params) in enumerate(
                zip(self.uploaded_ids, self.uploaded_groups, self.uploaded_models)):

            # Find the original sample count
            client_samples = 0
            for client in self.selected_clients:
                if client.id == client_id:
                    client_samples = client.train_samples
                    break

            group_params_dict[group_id].append(model_params)
            group_weights_dict[group_id].append(client_samples)

        # Aggregate for each group
        for group_id in group_params_dict.keys():
            params_list = group_params_dict[group_id]
            weights = group_weights_dict[group_id]

            # Normalize weights within group
            total_samples = sum(weights)
            normalized_weights = [w / total_samples for w in weights]

            # Aggregate
            aggregated = {}
            for key in params_list[0].keys():
                aggregated[key] = torch.zeros_like(params_list[0][key])

            for w, params in zip(normalized_weights, params_list):
                for key in aggregated.keys():
                    aggregated[key] += params[key] * w

            group_models[group_id] = aggregated
            print(f"  Group {group_id}: Aggregated {len(params_list)} models")

        return group_models

    def _aggregate_parameters_globally(self, keys):
        """Aggregate specified parameters globally"""
        aggregated = {}

        for key in keys:
            aggregated[key] = torch.zeros_like(self.uploaded_models[0][key])

        for w, model_params in zip(self.uploaded_weights, self.uploaded_models):
            for key in keys:
                aggregated[key] += model_params[key] * w

        return aggregated

    def _aggregate_parameters_by_groups(self, keys):
        """Aggregate specified parameters within groups"""
        group_params = {}

        # Group models by group ID
        group_params_dict = defaultdict(list)
        group_weights_dict = defaultdict(list)

        for i, (client_id, group_id, model_params) in enumerate(
                zip(self.uploaded_ids, self.uploaded_groups, self.uploaded_models)):

            client_samples = 0
            for client in self.selected_clients:
                if client.id == client_id:
                    client_samples = client.train_samples
                    break

            group_params_dict[group_id].append(model_params)
            group_weights_dict[group_id].append(client_samples)

        # Aggregate for each group
        for group_id in group_params_dict.keys():
            params_list = group_params_dict[group_id]
            weights = group_weights_dict[group_id]

            # Normalize weights within group
            total_samples = sum(weights)
            normalized_weights = [w / total_samples for w in weights]

            # Aggregate specified keys only
            aggregated = {}
            for key in keys:
                aggregated[key] = torch.zeros_like(params_list[0][key])

            for w, params in zip(normalized_weights, params_list):
                for key in keys:
                    aggregated[key] += params[key] * w

            group_params[group_id] = aggregated

        return group_params

    def _distribute_models(self, aggregated_model):
        """Distribute globally aggregated model to all clients"""
        for client in self.clients:
            model = load_item(client.role, 'model', client.save_folder_name)
            model.load_state_dict(aggregated_model)
            save_item(model, client.role, 'model', client.save_folder_name)

    def _distribute_group_models(self, group_models):
        """Distribute group-specific models to clients"""
        for client in self.clients:
            group_id = self.groups.get(client.id, 0)
            if group_id in group_models:
                model = load_item(client.role, 'model', client.save_folder_name)
                model.load_state_dict(group_models[group_id])
                save_item(model, client.role, 'model', client.save_folder_name)

    def _distribute_hybrid_models(self, global_params, global_keys, group_params, group_keys):
        """Distribute hybrid models (part global, part group)"""
        for client in self.clients:
            group_id = self.groups.get(client.id, 0)

            # Load current model
            model = load_item(client.role, 'model', client.save_folder_name)
            model_state = model.state_dict()

            # Update global part
            for key in global_keys:
                model_state[key] = global_params[key]

            # Update group part
            if group_id in group_params:
                for key in group_keys:
                    model_state[key] = group_params[group_id][key]

            # Load updated state
            model.load_state_dict(model_state)
            save_item(model, client.role, 'model', client.save_folder_name)

    def send_models(self):
        """Models are already distributed in aggregate_models_by_ratio"""
        pass