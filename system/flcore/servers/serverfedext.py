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
        self.args = args  # Store args for later use

        # FedEXT specific attributes
        self.groups = {}  # Dictionary to store client groups
        self.group_classifiers = {}  # Store group-level classifiers
        self.global_encoder = None  # Global encoder (FedAvg style)
        self.server_classifier = None  # Server's final classifier

        # Initialize client groups
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

            # Initialize global encoder and group classifiers
            sample_model = FedEXTModel(args, 0).to(self.device)
            self.global_encoder = copy.deepcopy(sample_model.encoder.state_dict())
            save_item(self.global_encoder, self.role, 'global_encoder', self.save_folder_name)

            # Initialize group classifiers
            for group_id in set(self.groups.values()):
                self.group_classifiers[group_id] = copy.deepcopy(sample_model.classifier.state_dict())
                save_item(self.group_classifiers[group_id], self.role, f'group_classifier_{group_id}', self.save_folder_name)

        print(f"Client groups: {self.groups}")
        print(f"Encoder ratio: {getattr(args, 'encoder_ratio', 0.7)}")

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

        print("\nStarting FedEXT training...")

        for i in range(1, self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # Send global encoder and group classifiers to selected clients
            self.send_models()

            print(f"\n-------------Round number: {i}-------------")
            print(f"Clients selected: {[c.id for c in self.selected_clients]}")

            # Train selected clients
            for client in self.selected_clients:
                client.train()

            # Receive models from clients
            self.receive_models()

            # Aggregate encoders globally (FedAvg style)
            if len(self.uploaded_encoders) > 0:
                self.aggregate_encoders()

            # Aggregate classifiers within groups
            if len(self.uploaded_classifiers) > 0:
                self.aggregate_group_classifiers()

            # Collect features and train server classifier in the last round
            if i == self.global_rounds:
                print("\n-------------Final Round: Training Server Classifier-------------")
                self.collect_features_and_train_server()

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
            final_acc = self.test_server_classifier()
            print(f"Final Server Accuracy: {final_acc:.4f}")

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
        """Send global encoder and group classifiers to selected clients"""
        assert (len(self.selected_clients) > 0)

        # Load global encoder
        global_encoder = load_item(self.role, 'global_encoder', self.save_folder_name)

        for client in self.selected_clients:
            # Get client's model
            model = load_item(client.role, 'model', client.save_folder_name)

            # Update encoder with global encoder
            model.encoder.load_state_dict(copy.deepcopy(global_encoder))

            # Update classifier with group classifier
            group_id = self.groups.get(client.id, 0)
            group_classifier = load_item(self.role, f'group_classifier_{group_id}', self.save_folder_name)
            model.classifier.load_state_dict(copy.deepcopy(group_classifier))

            # Save updated model
            save_item(model, client.role, 'model', client.save_folder_name)

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

            # Load client model
            model = load_item(client.role, 'model', client.save_folder_name)
            self.uploaded_encoders.append(model.encoder.state_dict())
            self.uploaded_classifiers.append(model.classifier.state_dict())

        # Normalize weights
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_encoders(self):
        """Aggregate encoders globally using FedAvg"""
        assert (len(self.uploaded_encoders) > 0)

        print("\nAggregating encoders globally (FedAvg style)...")

        # Initialize aggregated encoder
        aggregated_encoder = {}
        for key in self.uploaded_encoders[0].keys():
            aggregated_encoder[key] = torch.zeros_like(self.uploaded_encoders[0][key])

        # Weighted average of encoders
        for w, encoder_params in zip(self.uploaded_weights, self.uploaded_encoders):
            for key in aggregated_encoder.keys():
                aggregated_encoder[key] += encoder_params[key] * w

        # Save aggregated encoder
        self.global_encoder = aggregated_encoder
        save_item(self.global_encoder, self.role, 'global_encoder', self.save_folder_name)

        print(f"Aggregated {len(self.uploaded_encoders)} encoders globally")

    def aggregate_group_classifiers(self):
        """Aggregate classifiers within each group"""
        print("\nAggregating classifiers within groups...")

        # Group clients by their group ID
        group_classifiers_dict = defaultdict(list)
        group_weights_dict = defaultdict(list)

        for i, (client_id, group_id, classifier, weight) in enumerate(
                zip(self.uploaded_ids, self.uploaded_groups, self.uploaded_classifiers, self.uploaded_weights)):
            # Find the original sample count for this client
            client_samples = 0
            for client in self.selected_clients:
                if client.id == client_id:
                    client_samples = client.train_samples
                    break

            group_classifiers_dict[group_id].append(classifier)
            group_weights_dict[group_id].append(client_samples)

        # Aggregate classifiers for each group
        for group_id in group_classifiers_dict.keys():
            classifiers = group_classifiers_dict[group_id]
            weights = group_weights_dict[group_id]

            # Normalize weights within group
            total_samples = sum(weights)
            normalized_weights = [w / total_samples for w in weights]

            # Initialize aggregated classifier for this group
            aggregated_classifier = {}
            for key in classifiers[0].keys():
                aggregated_classifier[key] = torch.zeros_like(classifiers[0][key])

            # Weighted average within group
            for w, classifier_params in zip(normalized_weights, classifiers):
                for key in aggregated_classifier.keys():
                    aggregated_classifier[key] += classifier_params[key] * w

            self.group_classifiers[group_id] = aggregated_classifier
            save_item(aggregated_classifier, self.role, f'group_classifier_{group_id}', self.save_folder_name)

            print(f"Group {group_id}: Aggregated {len(classifiers)} classifiers")

    def collect_features_and_train_server(self):
        """Collect features from all clients and train server classifier"""
        print("\nCollecting features from all clients...")

        all_features = []
        all_labels = []

        # Collect features from all clients
        for client in self.clients:
            features, labels = client.extract_features_labels()
            all_features.append(features)
            all_labels.append(labels)

        # Concatenate all features and labels
        train_features = torch.cat(all_features, dim=0)
        train_labels = torch.cat(all_labels, dim=0)

        print(f"Collected {train_features.shape[0]} samples with feature dimension {train_features.shape[1]}")

        # Train server classifier
        print("\nTraining server classifier on collected features...")

        # Create server classifier
        self.server_classifier = nn.Linear(self.args.feature_dim, self.args.num_classes).to(self.device)

        # Create optimizer
        server_lr = getattr(self.args, 'server_learning_rate', 0.01)
        optimizer = torch.optim.SGD(self.server_classifier.parameters(), lr=server_lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Move data to device
        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)

        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        batch_size = getattr(self.args, 'server_batch_size', 32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Train server classifier
        server_epochs = getattr(self.args, 'server_epochs', 20)
        self.server_classifier.train()

        for epoch in range(server_epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()

                outputs = self.server_classifier(batch_features)
                loss = criterion(outputs, batch_labels)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            avg_loss = epoch_loss / len(dataloader)
            accuracy = correct / total

            if (epoch + 1) % 5 == 0:
                print(f"Server Epoch {epoch + 1}/{server_epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")

        # Save server classifier
        save_item(self.server_classifier, self.role, 'server_classifier', self.save_folder_name)
        print("Server classifier training completed")

    def test_server_classifier(self):
        """Test server classifier on global test dataset"""
        if self.server_classifier is None:
            return 0.0

        self.server_classifier.eval()

        # Use the global encoder for feature extraction
        global_encoder = load_item(self.role, 'global_encoder', self.save_folder_name)

        # Create a temporary model for feature extraction
        temp_model = FedEXTModel(self.args, 0).to(self.device)
        temp_model.encoder.load_state_dict(global_encoder)
        temp_model.eval()

        test_acc = 0
        test_num = 0

        # Test on all clients' test data
        for client in self.clients:
            testloader = client.load_test_data()

            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    # Extract features using global encoder
                    features = temp_model.encoder(x)

                    # Classify using server classifier
                    output = self.server_classifier(features)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

        accuracy = test_acc / test_num if test_num > 0 else 0

        print(f"Server Classifier: Acc: {accuracy:.4f}, Samples: {test_num}")

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "Server/test_accuracy": accuracy,
                "Server/test_samples": test_num,
                "Server/round": len(self.rs_test_acc)
            })

        return accuracy