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
import math


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

        # Initialize models with fine-grained layer splitting
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            # Initialize client models with improved FedEXT structure
            for client in self.clients:
                model = FedEXTModel(args, client.id).to(self.device)
                save_item(model, client.role, 'model', client.save_folder_name)
                
                # Set client's group
                if client.id in self.groups:
                    client.set_group(self.groups[client.id])
            
            # Configure and analyze model structure
            if len(self.clients) > 0:
                self._configure_and_analyze_models()
        
        print(f"\nClient groups: {self.groups}")
        
    def _initialize_groups(self):
        """Initialize client groups based on predefined assignment"""
        # Groups 0-4 with 4 clients each (hard-coded as requested)
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
    
    def _configure_and_analyze_models(self):
        """Configure fine-grained layer splitting for all models"""
        # Get a sample model to analyze structure
        sample_model = FedEXTModel(self.args, 0).to(self.device)
        
        # The model will automatically set split based on encoder_ratio in __init__
        # But we can also explicitly update it if needed
        if hasattr(sample_model, 'layers_list') and sample_model.layers_list:
            total_layers = len(sample_model.layers_list)
            
            # Calculate split index based on encoder_ratio
            if self.encoder_ratio >= 0.99:
                split_index = total_layers  # All global
            elif self.encoder_ratio <= 0.01:
                split_index = 0  # All local
            else:
                split_index = int(math.ceil(total_layers * self.encoder_ratio))
                # Ensure at least one layer is local (usually the head)
                split_index = min(split_index, total_layers - 1)
            
            # Apply split to all client models
            for client in self.clients:
                model = load_item(client.role, 'model', client.save_folder_name)
                model.set_layer_split(split_index)
                save_item(model, client.role, 'model', client.save_folder_name)
            
            # Store information for display
            self.total_layers = total_layers
            self.layer_split_index = split_index
            self.sample_model = sample_model
            self.sample_model.set_layer_split(split_index)
    
    def display_split_configuration(self):
        """Display detailed information about the model splitting configuration"""
        print("\n" + "="*90)
        print("                     FEDEXT MODEL SPLITTING CONFIGURATION")
        print("="*90)
        
        print(f"\n📊 Basic Configuration:")
        print(f"  • Encoder Ratio: {self.encoder_ratio:.2f}")
        print(f"  • Number of Clients: {self.num_clients}")
        print(f"  • Number of Groups: {len(set(self.groups.values()))}")
        
        if hasattr(self, 'sample_model') and self.sample_model.layers_list:
            total_layers = len(self.sample_model.layers_list)
            split_index = self.sample_model.layer_split_index
            
            print(f"\n🔧 Model Architecture:")
            print(f"  • Model Type: {self.sample_model.model_type}")
            print(f"  • Total Layers: {total_layers}")
            print(f"  • Split Point: Layer {split_index}")
            print(f"  • Global Layers: {split_index} ({split_index/total_layers*100:.1f}%)")
            print(f"  • Local Layers: {total_layers - split_index} ({(total_layers-split_index)/total_layers*100:.1f}%)")
            
            print(f"\n🌐 Aggregation Strategy:")
            if self.encoder_ratio >= 0.99:
                print(f"  ► Pure FedAvg Mode")
                print(f"    All layers will be globally aggregated across all clients")
            elif self.encoder_ratio <= 0.01:
                print(f"  ► Pure Group Mode") 
                print(f"    All layers will be aggregated only within groups")
            else:
                print(f"  ► Hybrid Mode")
                print(f"    • Layers 0-{split_index-1}: Global aggregation (FedAvg)")
                print(f"    • Layers {split_index}-{total_layers-1}: Group aggregation")
            
            # Display detailed layer information
            print(f"\n📋 Layer-wise Assignment:")
            print(f"  {'Index':<8} {'Layer Name':<35} {'Type':<25} {'Aggregation':<12} {'Params':<12}")
            print("  " + "-"*95)
            
            total_global_params = 0
            total_local_params = 0
            
            # Show first few and last few layers for brevity
            layers_to_show = []
            if total_layers <= 10:
                layers_to_show = list(range(total_layers))
            else:
                layers_to_show = list(range(5)) + ['...'] + list(range(total_layers-3, total_layers))
            
            for i in layers_to_show:
                if i == '...':
                    print(f"  {'...':<8} {'...':<35} {'...':<25} {'...':<12} {'...':<12}")
                    continue
                
                layer_name, layer = self.sample_model.layers_list[i]
                layer_type = layer.__class__.__name__
                
                # Shorten long names for display
                if len(layer_name) > 33:
                    display_name = layer_name[:30] + "..."
                else:
                    display_name = layer_name
                
                if len(layer_type) > 23:
                    display_type = layer_type[:20] + "..."
                else:
                    display_type = layer_type
                
                # Count parameters
                num_params = sum(p.numel() for p in layer.parameters())
                
                # Determine aggregation type
                if i < split_index:
                    agg_type = "🌍 Global"
                    total_global_params += num_params
                else:
                    agg_type = "👥 Group"
                    total_local_params += num_params
                
                param_str = f"{num_params:,}"
                if len(param_str) > 10:
                    param_str = f"{num_params/1e6:.1f}M"
                
                print(f"  [{i:^6}] {display_name:<35} {display_type:<25} {agg_type:<12} {param_str:<12}")
            
            # If we skipped some layers, calculate their parameters
            if '...' in layers_to_show:
                for i in range(5, total_layers-3):
                    layer_name, layer = self.sample_model.layers_list[i]
                    num_params = sum(p.numel() for p in layer.parameters())
                    if i < split_index:
                        total_global_params += num_params
                    else:
                        total_local_params += num_params
            
            # Parameter distribution summary
            total_params = total_global_params + total_local_params
            if total_params > 0:
                print("\n  " + "-"*95)
                print(f"  📊 Parameter Distribution:")
                print(f"    • Global parameters: {total_global_params:,} ({total_global_params/total_params*100:.1f}%)")
                print(f"    • Local parameters:  {total_local_params:,} ({total_local_params/total_params*100:.1f}%)")
                print(f"    • Total parameters:  {total_params:,}")
        
        # Display group assignments
        print(f"\n👥 Client Group Assignments:")
        group_members = defaultdict(list)
        for client_id, group_id in self.groups.items():
            group_members[group_id].append(client_id)
        
        for group_id in sorted(group_members.keys()):
            members = sorted(group_members[group_id])
            print(f"  • Group {group_id}: Clients {members} ({len(members)} members)")
        
        print("\n" + "="*90 + "\n")
    
    def train(self):
        # Display detailed split configuration at the start
        self.display_split_configuration()
        
        # Initial evaluation
        print(f"\n-------------Initial Evaluation-------------")
        print("\nEvaluate initial models performance")
        self.evaluate()
        
        print(f"\nStarting FedEXT training with fine-grained layer splitting...")
        
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
            
            # Aggregate models using fine-grained layer splitting
            self.aggregate_models_layer_wise()
            
            # Send updated models back to clients
            self.send_models()
            
            # Evaluation
            if i % self.eval_gap == 0:
                print("\nEvaluate models after aggregation")
                self.evaluate()
            
            self.Budget.append(time.time() - s_t)
            print('-'*50, 'Round time cost:', self.Budget[-1])
            
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        
        # Train and test global model after final round
        if 0.01 < self.encoder_ratio < 0.99:
            print(f"\n-------------Building Global Model-------------")
            self.build_and_test_global_model()
        
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
            self.uploaded_models.append(model)
        
        # Normalize weights
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    
    def aggregate_models_layer_wise(self):
        """
        Aggregate models using fine-grained layer-wise splitting.
        Uses the model's built-in get_global_params() and get_local_params() methods.
        """
        print(f"\nAggregating models with layer-wise splitting (encoder_ratio={self.encoder_ratio:.2f})...")
        
        # Get global and local parameters from each uploaded model
        global_params_list = []
        local_params_by_group = defaultdict(list)
        weights_by_group = defaultdict(list)
        
        for i, (model, weight, group_id, client_id) in enumerate(
                zip(self.uploaded_models, self.uploaded_weights, self.uploaded_groups, self.uploaded_ids)):
            
            # Extract global parameters (for FedAvg aggregation)
            global_params = model.get_global_params()
            if global_params:  # Only if there are global parameters
                global_params_list.append((global_params, weight))
            
            # Extract local parameters (for group aggregation)
            local_params = model.get_local_params()
            if local_params:  # Only if there are local parameters
                # Find original sample count for proper weighting within group
                client_samples = 0
                for client in self.selected_clients:
                    if client.id == client_id:
                        client_samples = client.train_samples
                        break
                
                local_params_by_group[group_id].append(local_params)
                weights_by_group[group_id].append(client_samples)
        
        # Aggregate global parameters (FedAvg)
        aggregated_global = None
        if global_params_list:
            print(f"  Aggregating global parameters from {len(global_params_list)} clients...")
            aggregated_global = self._fedavg_aggregation(global_params_list)
            print(f"    Aggregated {len(aggregated_global)} global parameters")
        else:
            print(f"  No global parameters to aggregate (pure group mode)")
        
        # Aggregate local parameters by group
        aggregated_local_by_group = {}
        if local_params_by_group:
            print(f"  Aggregating local parameters for {len(local_params_by_group)} groups...")
            for group_id, params_list in local_params_by_group.items():
                # Normalize weights within group
                group_weights = weights_by_group[group_id]
                total_samples = sum(group_weights)
                normalized_weights = [w / total_samples for w in group_weights]
                
                # Create weighted params list
                weighted_params = [(params, w) for params, w in zip(params_list, normalized_weights)]
                aggregated_local_by_group[group_id] = self._fedavg_aggregation(weighted_params)
                
                print(f"    Group {group_id}: Aggregated {len(params_list)} models, "
                      f"{len(aggregated_local_by_group[group_id])} parameters")
        else:
            print(f"  No local parameters to aggregate (pure FedAvg mode)")
        
        # Distribute aggregated parameters back to clients
        self._distribute_layer_wise_models(aggregated_global, aggregated_local_by_group)
    
    def _fedavg_aggregation(self, weighted_params_list):
        """
        Perform FedAvg aggregation on a list of weighted parameters.
        
        Args:
            weighted_params_list: List of (params_dict, weight) tuples
        Returns:
            Aggregated parameters dictionary
        """
        if not weighted_params_list:
            return {}
        
        # Initialize with zeros
        aggregated = OrderedDict()
        first_params = weighted_params_list[0][0]
        
        for key in first_params.keys():
            aggregated[key] = torch.zeros_like(first_params[key])
        
        # Weighted sum
        for params, weight in weighted_params_list:
            for key in params.keys():
                if key in aggregated:
                    aggregated[key] += params[key] * weight
        
        return aggregated
    
    def _distribute_layer_wise_models(self, global_params, local_params_by_group):
        """
        Distribute aggregated parameters to selected clients using model's 
        set_global_params() and set_local_params() methods.
        """
        for client in self.selected_clients:
            group_id = self.groups.get(client.id, 0)
            
            # Load client model
            model = load_item(client.role, 'model', client.save_folder_name)
            
            # Update global parameters if available
            if global_params:
                model.set_global_params(global_params)
            
            # Update local parameters for the client's group
            if group_id in local_params_by_group:
                model.set_local_params(local_params_by_group[group_id])
            
            # Save updated model
            save_item(model, client.role, 'model', client.save_folder_name)
    
    def send_models(self):
        """Models are already distributed in aggregate_models_layer_wise"""
        pass
    
    def build_and_test_global_model(self):
        """
        Build and test a global model by:
        1. Collecting embeddings from all clients using the globally aggregated encoder
        2. Training a global classifier (head) on server
        3. Testing on balanced global test set
        """
        print("\nCollecting embeddings from all clients...")
        
        # Collect training embeddings and labels from all clients
        train_embeddings_list = []
        train_labels_list = []
        
        for client in self.clients:
            features, labels = client.extract_features_labels()
            train_embeddings_list.append(features)
            train_labels_list.append(labels)
        
        # Concatenate all training data
        train_embeddings = torch.cat(train_embeddings_list, dim=0)
        train_labels = torch.cat(train_labels_list, dim=0)
        
        print(f"Collected {train_embeddings.shape[0]} training samples with embedding dimension {train_embeddings.shape[1]}")
        
        # Build global test dataset with balanced sampling
        print("\nBuilding balanced global test dataset...")
        test_embeddings_list = []
        test_labels_list = []
        
        # Collect test embeddings from all clients
        for client in self.clients:
            test_features, test_labels = client.extract_test_features_labels()
            test_embeddings_list.append(test_features)
            test_labels_list.append(test_labels)
        
        # Balance test set - take equal samples from each client
        min_test_samples = min([emb.shape[0] for emb in test_embeddings_list])
        balanced_test_embeddings = []
        balanced_test_labels = []
        
        for emb, lab in zip(test_embeddings_list, test_labels_list):
            # Randomly sample min_test_samples from each client
            indices = torch.randperm(emb.shape[0])[:min_test_samples]
            balanced_test_embeddings.append(emb[indices])
            balanced_test_labels.append(lab[indices])
        
        test_embeddings = torch.cat(balanced_test_embeddings, dim=0)
        test_labels = torch.cat(balanced_test_labels, dim=0)
        
        print(f"Built balanced test set with {test_embeddings.shape[0]} samples ({min_test_samples} per client)")
        
        # Train global classifier (head) on server
        print("\nTraining global classifier on server...")
        
        # Get sample model to understand the local layers structure
        sample_model = FedEXTModel(self.args, 0).to(self.device)
        sample_model.set_layer_split(self.layer_split_index)
        
        # Create a classifier from the local layers
        if sample_model.local_layers and len(sample_model.local_layers) > 0:
            # Build a sequential classifier from local layers
            local_modules = []
            for layer_name, layer in sample_model.local_layers:
                # Clone the layer to create a fresh classifier
                local_modules.append(copy.deepcopy(layer))
            
            if len(local_modules) > 1:
                global_classifier = nn.Sequential(*local_modules)
            else:
                global_classifier = local_modules[0]
            
            print(f"Training classifier with {len(local_modules)} local layers")
        else:
            # No local layers (pure FedAvg), create a simple linear classifier
            feature_dim = train_embeddings.shape[1]
            num_classes = self.args.num_classes
            global_classifier = nn.Linear(feature_dim, num_classes)
            print(f"Training new linear classifier (all layers are global)")
        
        # Move to device
        global_classifier = global_classifier.to(self.device)
        train_embeddings = train_embeddings.to(self.device)
        train_labels = train_labels.to(self.device)
        test_embeddings = test_embeddings.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Create data loader for training
        train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        
        # Train the global classifier
        optimizer = torch.optim.SGD(
            global_classifier.parameters(), 
            lr=self.args.local_learning_rate,
            momentum=0.9,
            weight_decay=1e-4
        )
        criterion = nn.CrossEntropyLoss()
        
        global_classifier.train()
        for epoch in range(10):  # Train for 10 epochs
            epoch_loss = 0
            correct = 0
            total = 0
            
            for features, labels in train_loader:
                optimizer.zero_grad()
                outputs = global_classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = correct / total
            if epoch % 2 == 0:
                print(f"  Epoch {epoch}: Loss = {epoch_loss/len(train_loader):.4f}, Train Acc = {train_acc:.4f}")
        
        # Test global model
        print("\nTesting global model...")
        global_classifier.eval()
        
        with torch.no_grad():
            outputs = global_classifier(test_embeddings)
            _, predictions = torch.max(outputs, 1)
            accuracy = (predictions == test_labels).float().mean().item()
        
        print(f"Global Model Test Accuracy: {accuracy:.4f}")
        print(f"  (Tested on {test_embeddings.shape[0]} balanced samples)")
        
        # Store global model results
        self.global_test_acc = accuracy
        
        return accuracy
    
    def adaptive_update_split_ratio(self, new_ratio):
        """
        Dynamically update the split ratio for all clients during training.
        Useful for adaptive splitting strategies.
        
        Args:
            new_ratio: New encoder ratio between 0 and 1
        """
        print(f"\n📊 Updating split ratio from {self.encoder_ratio:.2f} to {new_ratio:.2f}")
        
        self.encoder_ratio = new_ratio
        
        # Update all client models
        for client in self.clients:
            model = load_item(client.role, 'model', client.save_folder_name)
            model.update_split_ratio(new_ratio)
            save_item(model, client.role, 'model', client.save_folder_name)
        
        # Update sample model for display
        if hasattr(self, 'sample_model'):
            self.sample_model.update_split_ratio(new_ratio)
            self.layer_split_index = self.sample_model.layer_split_index
        
        print(f"  Updated all {len(self.clients)} client models")
        print(f"  New split: {self.layer_split_index}/{self.total_layers} layers are global")