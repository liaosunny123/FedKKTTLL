import time
import numpy as np
import torch
import torch.nn as nn
import copy
import json
import os
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
        self.args = args  # Store args for later use

        # FedEXT specific attributes
        self.groups = {}  # Dictionary to store client groups
        self.group_classifiers = {}  # Store group-level classifiers
        self.server_classifier = None  # Server's final classifier
        self.collected_features = []  # Store collected features from clients
        self.collected_labels = []  # Store collected labels from clients

        # Initialize client groups from distribution config
        self._initialize_groups()

        # Initialize models
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            # Initialize client models with FedEXT structure
            for client in self.clients:
                model = FedEXTModel(args, client.id).to(self.device)
                save_item(model, client.role, 'model', client.save_folder_name)

                # Set client's group
                if client.id in self.groups:
                    group_id = self.groups[client.id]
                    client.set_group(group_id)

        print(f"Client groups: {self.groups}")

    def _initialize_groups(self):
        """Initialize client groups from distribution config"""
        if self.args.distribution_config and os.path.exists(self.args.distribution_config):
            with open(self.args.distribution_config, 'r') as f:
                config = json.load(f)

            for client_id, client_config in config.items():
                if 'group' in client_config:
                    self.groups[int(client_id)] = client_config['group']
                else:
                    # Default group assignment if not specified
                    self.groups[int(client_id)] = 0

            print(f"Loaded client groups from distribution config: {self.groups}")
        else:
            # Default: all clients in one group
            for client in self.clients:
                self.groups[client.id] = 0
            print(f"No distribution config found, all clients assigned to group 0")

    def train(self):
        # Initial evaluation
        print(f"\n-------------Initial Evaluation-------------")
        print("\nEvaluate initial models performance")
        self.evaluate()

        print("\nStarting FedEXT training...")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n-------------Round number: {i}-------------")
            print(f"Clients selected: {[c.id for c in self.selected_clients]}")

            # Train selected clients
            for client in self.selected_clients:
                client.train()

            # Receive and aggregate models
            self.receive_models()

            # Aggregate encoders (FedAvg style)
            self.aggregate_encoders()

            # Aggregate classifiers within groups
            self.aggregate_group_classifiers()

            # Send updated models back to clients
            self.send_aggregated_models()

            # In the last round, collect features and train server classifier
            if i == self.global_rounds:
                print("\n-------------Final Round: Collecting Features-------------")
                self.collect_features_labels()
                self.train_server_classifier()

            # Evaluation
            if i % self.eval_gap == 0:
                print("\nEvaluate models after aggregation")
                self.evaluate()

                # Test server classifier if it exists
                if self.server_classifier is not None:
                    print("\nEvaluate server classifier performance")
                    self.test_server_classifier()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, 'Round time cost:', self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy:")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round:")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else 0)

        # Final evaluation with server classifier
        if self.server_classifier is not None:
            print("\n-------------Final Server Classifier Evaluation-------------")
            self.test_server_classifier()

        # Log final metrics to wandb
        if self.use_wandb:
            wandb.log({
                "Final/best_accuracy": max(self.rs_test_acc),
                "Final/best_auc": max(self.rs_test_auc) if self.rs_test_auc else 0,
                "Final/avg_time_per_round": sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else 0,
                "Final/total_rounds": len(self.rs_test_acc)
            })
            wandb.finish()

        self.save_results()

    def send_models(self):
        """Initial model sending (if needed)"""
        pass  # In FedEXT, we don't send global model initially

    def receive_models(self):
        """Receive encoder and classifier parameters from selected clients"""
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_encoders = []
        self.uploaded_classifiers = []
        self.uploaded_groups = []
        tot_samples = 0

        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_groups.append(self.groups.get(client.id, 0))

            # Get encoder and classifier parameters
            encoder_params = client.get_encoder_params()
            classifier_params = client.get_classifier_params()

            self.uploaded_encoders.append(encoder_params)
            self.uploaded_classifiers.append(classifier_params)

        # Normalize weights
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_encoders(self):
        """Aggregate encoders using FedAvg across all clients"""
        assert (len(self.uploaded_encoders) > 0)

        print("\nAggregating encoders across all clients...")

        # Initialize aggregated encoder parameters
        aggregated_encoder = {}
        for key in self.uploaded_encoders[0].keys():
            aggregated_encoder[key] = torch.zeros_like(self.uploaded_encoders[0][key])

        # Weighted average of all encoders
        for w, encoder_params in zip(self.uploaded_weights, self.uploaded_encoders):
            for key in aggregated_encoder.keys():
                if key in encoder_params:
                    aggregated_encoder[key] += encoder_params[key] * w

        # Store aggregated encoder
        self.aggregated_encoder = aggregated_encoder
        print(f"Aggregated {len(self.uploaded_encoders)} client encoders")

    def aggregate_group_classifiers(self):
        """Aggregate classifiers within each group"""
        print("\nAggregating classifiers within groups...")

        # Group clients by their group ID
        group_classifiers_dict = defaultdict(list)
        group_weights_dict = defaultdict(list)

        for i, (group_id, classifier, weight) in enumerate(
                zip(self.uploaded_groups, self.uploaded_classifiers, self.uploaded_weights)):
            group_classifiers_dict[group_id].append(classifier)
            group_weights_dict[group_id].append(weight)

        # Aggregate classifiers for each group
        self.group_classifiers = {}
        for group_id in group_classifiers_dict.keys():
            classifiers = group_classifiers_dict[group_id]
            weights = group_weights_dict[group_id]

            # Normalize weights within group
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Initialize aggregated classifier for this group
            aggregated_classifier = {}
            for key in classifiers[0].keys():
                aggregated_classifier[key] = torch.zeros_like(classifiers[0][key])

            # Weighted average within group
            for w, classifier_params in zip(normalized_weights, classifiers):
                for key in aggregated_classifier.keys():
                    if key in classifier_params:
                        aggregated_classifier[key] += classifier_params[key] * w

            self.group_classifiers[group_id] = aggregated_classifier
            print(f"Group {group_id}: Aggregated {len(classifiers)} classifiers")

    def send_aggregated_models(self):
        """Send aggregated encoder and group-specific classifier back to clients"""
        for client in self.selected_clients:
            # Send aggregated encoder to all clients
            client.receive_encoder(copy.deepcopy(self.aggregated_encoder))

            # Send group-specific classifier
            group_id = self.groups.get(client.id, 0)
            if group_id in self.group_classifiers:
                client.receive_classifier(copy.deepcopy(self.group_classifiers[group_id]))

    def collect_features_labels(self):
        """Collect features and labels from all clients for server training"""
        print("\nCollecting features and labels from all clients...")

        self.collected_features = []
        self.collected_labels = []

        for client in self.clients:
            features, labels = client.extract_features_labels()
            self.collected_features.append(features)
            self.collected_labels.append(labels)

        # Concatenate all features and labels
        self.collected_features = torch.cat(self.collected_features, dim=0)
        self.collected_labels = torch.cat(self.collected_labels, dim=0)

        print(f"Collected {self.collected_features.shape[0]} samples with feature dimension {self.collected_features.shape[1]}")

    def train_server_classifier(self):
        """Train server's own classifier on collected features"""
        print("\nTraining server classifier on collected features...")

        # Create server classifier
        self.server_classifier = nn.Linear(self.args.feature_dim, self.args.num_classes).to(self.device)

        # Create optimizer
        optimizer = torch.optim.SGD(self.server_classifier.parameters(),
                                     lr=self.args.server_learning_rate if hasattr(self.args, 'server_learning_rate') else 0.01)
        criterion = nn.CrossEntropyLoss()

        # Move data to device
        features = self.collected_features.to(self.device)
        labels = self.collected_labels.to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(features, labels)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=self.args.server_batch_size if hasattr(self.args, 'server_batch_size') else 32,
                                                  shuffle=True)

        # Train server classifier
        server_epochs = self.args.server_epochs if hasattr(self.args, 'server_epochs') else 50
        self.server_classifier.train()

        for epoch in range(server_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()

                outputs = self.server_classifier(batch_features)
                loss = criterion(outputs, batch_labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            if (epoch + 1) % 10 == 0:
                print(f"Server Classifier - Epoch {epoch + 1}/{server_epochs}, Loss: {avg_loss:.4f}")

        print("Server classifier training completed")

    def test_server_classifier(self):
        """Test server classifier on global test dataset"""
        if self.server_classifier is None:
            return None, None, None

        self.server_classifier.eval()

        # Test on aggregated test dataset
        test_acc = 0
        test_num = 0

        # Get a sample client model to use its encoder
        sample_client = self.clients[0]
        model = load_item(sample_client.role, 'model', sample_client.save_folder_name)
        model.to(self.device)
        model.eval()

        for client in self.clients:
            testloader = client.load_test_data()

            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    # Extract features using encoder
                    features = model.encoder(x)

                    # Classify using server classifier
                    output = self.server_classifier(features)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

        accuracy = test_acc / test_num if test_num > 0 else 0

        print(f"Server Classifier: Acc: {accuracy:.4f}, Samples: {test_num}")

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "Server/classifier_test_accuracy": accuracy,
                "Server/classifier_test_samples": test_num,
                "Server/round": len(self.rs_test_acc)
            })

        return test_acc, test_num, accuracy