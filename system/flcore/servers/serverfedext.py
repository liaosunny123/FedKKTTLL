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
from collections import defaultdict


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

        # Initialize models
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            # Initialize client models with FedEXT structure
            for client in self.clients:
                model = FedEXTModel(args, client.id).to(self.device)
                save_item(model, client.role, 'model', client.save_folder_name)

                # Set client's group
                if client.id in self.groups:
                    client.set_group(self.groups[client.id])

            # Analyze model structure for the first client
            if len(self.clients) > 0:
                sample_model = FedEXTModel(args, 0).to(self.device)
                base_params = sample_model.get_base_params()
                head_params = sample_model.get_head_params()
                total_params = dict(sample_model.state_dict())

                base_keys = len(base_params.keys())
                head_keys = len(head_params.keys())
                total_keys = len(total_params.keys())

                print(f"\nModel structure analysis:")
                print(f"  Total parameter keys: {total_keys}")
                print(f"  Base (encoder) keys: {base_keys} ({base_keys/total_keys*100:.1f}%)")
                print(f"  Head (classifier) keys: {head_keys} ({head_keys/total_keys*100:.1f}%)")

        print(f"\nClient groups: {self.groups}")
        print(f"Encoder ratio: {self.encoder_ratio}")
        print(f"Aggregation strategy based on encoder_ratio:")
        if self.encoder_ratio >= 0.99:
            print(f"  → Pure FedAvg mode (all parameters global aggregation)")
        elif self.encoder_ratio <= 0.01:
            print(f"  → Pure Group mode (all parameters group aggregation)")
        else:
            print(f"  → Hybrid mode:")
            print(f"    - First {self.encoder_ratio*100:.0f}% of parameters: global aggregation")
            print(f"    - Last {(1-self.encoder_ratio)*100:.0f}% of parameters: group aggregation")

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
        Split parameters based on ratio and aggregate accordingly
        """
        print(f"\nAggregating models (encoder_ratio={self.encoder_ratio})...")

        # Get all parameter keys
        all_keys = list(self.uploaded_models[0].keys())
        total_keys = len(all_keys)

        # Calculate split point based on encoder_ratio
        encoder_keys_count = int(total_keys * self.encoder_ratio)

        # Split keys into encoder and classifier
        encoder_keys = all_keys[:encoder_keys_count]
        classifier_keys = all_keys[encoder_keys_count:]

        print(f"  Encoder parameters (global): {len(encoder_keys)} keys")
        print(f"  Classifier parameters (group): {len(classifier_keys)} keys")

        # Special cases
        if self.encoder_ratio >= 0.99:
            # Pure FedAvg: aggregate all parameters globally
            print("  Mode: Pure FedAvg (all global aggregation)")
            aggregated_models = self._aggregate_all_globally()
            self._distribute_models(aggregated_models)

        elif self.encoder_ratio <= 0.01:
            # Pure Group: aggregate all parameters within groups
            print("  Mode: Pure Group (all group aggregation)")
            group_models = self._aggregate_all_by_groups()
            self._distribute_group_models(group_models)

        else:
            # Hybrid: encoder global, classifier group
            print("  Mode: Hybrid (mixed aggregation)")

            # Aggregate encoder globally
            global_encoder = self._aggregate_parameters_globally(encoder_keys)

            # Aggregate classifier by groups
            group_classifiers = self._aggregate_parameters_by_groups(classifier_keys)

            # Combine and distribute
            self._distribute_hybrid_models(global_encoder, encoder_keys, group_classifiers, classifier_keys)

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

    def _distribute_hybrid_models(self, global_encoder, encoder_keys, group_classifiers, classifier_keys):
        """Distribute hybrid models (global encoder + group classifier) to clients"""
        for client in self.clients:
            group_id = self.groups.get(client.id, 0)

            # Load current model
            model = load_item(client.role, 'model', client.save_folder_name)
            model_state = model.state_dict()

            # Update encoder parameters (global)
            for key in encoder_keys:
                model_state[key] = global_encoder[key]

            # Update classifier parameters (from group)
            if group_id in group_classifiers:
                for key in classifier_keys:
                    model_state[key] = group_classifiers[group_id][key]

            # Load updated state
            model.load_state_dict(model_state)
            save_item(model, client.role, 'model', client.save_folder_name)

    def send_models(self):
        """Models are already distributed in aggregate_models_by_ratio"""
        pass