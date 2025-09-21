import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import wandb
from collections import OrderedDict


class clientFedEXT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        # Learning rate settings (same as FedAvg)
        self.learning_rate_decay = getattr(args, 'learning_rate_decay', False)
        self.learning_rate_decay_gamma = getattr(args, 'learning_rate_decay_gamma', 0.99)
        self.current_round = 0
        
        # FedEXT specific attributes
        self.group_id = None  # Will be set by server
        self.encoder_ratio = getattr(args, 'encoder_ratio', 0.7)
        
        # Track model split information
        self.split_info = None
        self.global_param_count = 0
        self.local_param_count = 0
    
    def set_group(self, group_id):
        """Set the group ID for this client and report model configuration"""
        self.group_id = group_id
        
        # Load model to get split information
        model = load_item(self.role, 'model', self.save_folder_name)
        
        # Get and store split information
        if hasattr(model, 'get_split_info'):
            self.split_info = model.get_split_info()
            
            # Count parameters in global vs local parts
            self.global_param_count = self._count_params_in_dict(model.get_global_params())
            self.local_param_count = self._count_params_in_dict(model.get_local_params())
            
            total_params = self.global_param_count + self.local_param_count
            if total_params > 0:
                global_ratio = self.global_param_count / total_params * 100
                local_ratio = self.local_param_count / total_params * 100
            else:
                global_ratio = local_ratio = 0
            
            print(f"\n📌 Client {self.id} Configuration:")
            print(f"  • Group ID: {self.group_id}")
            print(f"  • Model type: {model.model_type if hasattr(model, 'model_type') else 'unknown'}")
            print(f"  • Layer split: {self.split_info['split_index']}/{self.split_info['total_layers']} layers are global")
            print(f"  • Global params: {self.global_param_count:,} ({global_ratio:.1f}%)")
            print(f"  • Local params: {self.local_param_count:,} ({local_ratio:.1f}%)")
            print(f"  • Encoder ratio: {self.split_info['encoder_ratio']:.2f}")
        else:
            print(f"Client {self.id} assigned to group {self.group_id}")
        
        save_item(model, self.role, 'model', self.save_folder_name)
    
    def _count_params_in_dict(self, param_dict):
        """Count total number of parameters in a parameter dictionary"""
        if not param_dict:
            return 0
        return sum(p.numel() for p in param_dict.values())
    
    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        
        # Update current round
        self.current_round += 1
        
        # Apply learning rate decay
        current_lr = self.learning_rate
        if self.learning_rate_decay:
            current_lr = self.learning_rate * (self.learning_rate_decay_gamma ** self.current_round)
        
        # Create separate optimizers for global and local parameters if needed
        if hasattr(model, 'global_layers') and hasattr(model, 'local_layers'):
            # Get parameter groups
            global_params = []
            local_params = []
            
            for layer_name, layer in model.global_layers:
                global_params.extend(layer.parameters())
            
            for layer_name, layer in model.local_layers:
                local_params.extend(layer.parameters())
            
            param_groups = []
            if global_params:
                param_groups.append({'params': global_params, 'lr': current_lr})
            if local_params:
                local_lr = current_lr 
                param_groups.append({'params': local_params, 'lr': local_lr})
            
            if param_groups:
                optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=1e-4)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9, weight_decay=1e-4)
        else:
            # Fallback to standard optimizer
            optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, momentum=0.9, weight_decay=1e-4)
        
        model.train()
        
        start_time = time.time()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        epoch_losses = []
        epoch_accuracies = []
        
        for epoch in range(max_local_epochs):
            batch_losses = []
            correct = 0
            total = 0
            
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                # Forward pass
                output = model(x)
                loss = self.loss(output, y)
                batch_losses.append(loss.item())
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
            
            # Calculate epoch metrics
            if batch_losses:
                epoch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(epoch_loss)
                epoch_acc = correct / total if total > 0 else 0
                epoch_accuracies.append(epoch_acc)
                
                # Log epoch metrics to wandb
                if self.use_wandb:
                    global_step = self.current_round * 1000 + epoch
                    wandb.log({
                        f"Client_{self.id}/epoch_loss": epoch_loss,
                        f"Client_{self.id}/epoch_accuracy": epoch_acc,
                        f"Client_{self.id}/learning_rate": current_lr,
                        f"Client_{self.id}/round": self.current_round,
                        f"Client_{self.id}/local_epoch": epoch,
                        f"Client_{self.id}/group_id": self.group_id,
                    }, step=global_step)
        
        save_item(model, self.role, 'model', self.save_folder_name)
        
        train_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += train_time
        
        # Calculate training summary
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        avg_acc = sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0
        
        # Log training summary to wandb
        if self.use_wandb:
            global_step = self.current_round * 1000 + max_local_epochs
            wandb.log({
                f"Client_{self.id}/train_time": train_time,
                f"Client_{self.id}/total_epochs": max_local_epochs,
                f"Client_{self.id}/avg_train_loss": avg_loss,
                f"Client_{self.id}/avg_train_accuracy": avg_acc,
                f"Client_{self.id}/global_round": self.current_round,
                f"Client_{self.id}/num_train_samples": len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0,
                f"Client_{self.id}/global_param_count": self.global_param_count,
                f"Client_{self.id}/local_param_count": self.local_param_count,
            }, step=global_step)
        
        # Print training information with more details
        num_samples = len(trainloader.dataset.indices) if hasattr(trainloader.dataset, 'indices') else len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 'unknown'
        print(f"Client {self.id} (Group {self.group_id}): "
              f"Round {self.current_round} - "
              f"Trained on {num_samples} samples, "
              f"Avg loss: {avg_loss:.4f}, "
              f"Avg acc: {avg_acc:.4f}")
    
    def extract_features_labels(self):
        """
        Extract features and labels from local training data using the globally aggregated layers.
        This is called to collect embeddings for server-side global classifier training.
        """
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # Extract features using globally aggregated layers
                features = model.extract_global_features(x)
                
                all_features.append(features.cpu())
                all_labels.append(y.cpu())
        
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        print(f"Client {self.id} (Group {self.group_id}): "
              f"Extracted {features.shape[0]} features with dimension {features.shape[1]}")
        
        return features, labels
    
    def extract_test_features_labels(self):
        """
        Extract features and labels from local test data.
        Used for building global test dataset.
        """
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # Extract features using globally aggregated layers
                features = model.extract_global_features(x)
                
                all_features.append(features.cpu())
                all_labels.append(y.cpu())
        
        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        return features, labels
    
    def get_model_info(self):
        """
        Get detailed information about the model's current state and splitting configuration.
        Useful for debugging and monitoring.
        """
        model = load_item(self.role, 'model', self.save_folder_name)
        
        info = {
            'client_id': self.id,
            'group_id': self.group_id,
            'current_round': self.current_round,
        }
        
        if hasattr(model, 'get_split_info'):
            split_info = model.get_split_info()
            info.update(split_info)
        
        # Get parameter statistics
        if hasattr(model, 'get_global_params') and hasattr(model, 'get_local_params'):
            global_params = model.get_global_params()
            local_params = model.get_local_params()
            
            info['global_param_keys'] = len(global_params)
            info['local_param_keys'] = len(local_params)
            info['global_param_count'] = self._count_params_in_dict(global_params)
            info['local_param_count'] = self._count_params_in_dict(local_params)
            
            # Calculate norms for monitoring training health
            global_norm = torch.norm(torch.cat([p.flatten() for p in global_params.values()])).item() if global_params else 0
            local_norm = torch.norm(torch.cat([p.flatten() for p in local_params.values()])).item() if local_params else 0
            
            info['global_param_norm'] = global_norm
            info['local_param_norm'] = local_norm
        
        return info
    
    def test_metrics(self):
        """
        Test the model and calculate metrics.
        Enhanced with split-aware testing capabilities.
        """
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)
        model.eval()
        
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        # Track separate accuracies if using split model
        features_collected = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                # Standard forward pass
                output = model(x)
                
                # Also collect intermediate features if available
                if hasattr(model, 'extract_global_features'):
                    features = model.extract_global_features(x)
                    features_collected.append(features.cpu())
                
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)
        
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        
        # Calculate AUC with softmax normalization
        y_prob_softmax = torch.softmax(torch.from_numpy(y_prob), dim=1).numpy()
        auc = metrics.roc_auc_score(y_true, y_prob_softmax, average='micro')
        
        test_accuracy = test_acc / test_num if test_num > 0 else 0
        
        # Calculate feature statistics if collected
        feature_stats = {}
        if features_collected:
            all_features = torch.cat(features_collected, dim=0)
            feature_stats = {
                'feature_dim': all_features.shape[1],
                'feature_mean': all_features.mean().item(),
                'feature_std': all_features.std().item(),
                'feature_norm': torch.norm(all_features, dim=1).mean().item()
            }
        
        # Log test metrics to wandb
        if self.use_wandb:
            log_dict = {
                f"Client_{self.id}/test_accuracy": test_accuracy,
                f"Client_{self.id}/test_auc": auc,
                f"Client_{self.id}/test_samples": test_num,
                f"Client_{self.id}/group_id": self.group_id,
            }
            
            # Add feature statistics if available
            if feature_stats:
                for key, value in feature_stats.items():
                    log_dict[f"Client_{self.id}/{key}"] = value
            
            wandb.log(log_dict)
        
        # Print detailed test results
        print(f"Client {self.id} (Group {self.group_id}) Test Results: "
              f"Acc={test_accuracy:.4f}, AUC={auc:.4f}, Samples={test_num}")
        
        if feature_stats:
            print(f"  Feature stats: dim={feature_stats['feature_dim']}, "
                  f"norm={feature_stats['feature_norm']:.3f}")
        
        return test_acc, test_num, auc
    
    def save_model_checkpoint(self, path_suffix=""):
        """
        Save a checkpoint of the current model with split information.
        Useful for analysis and recovery.
        """
        model = load_item(self.role, 'model', self.save_folder_name)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'client_id': self.id,
            'group_id': self.group_id,
            'current_round': self.current_round,
            'encoder_ratio': self.encoder_ratio
        }
        
        if hasattr(model, 'get_split_info'):
            checkpoint['split_info'] = model.get_split_info()
        
        # Save global and local params separately for analysis
        if hasattr(model, 'get_global_params'):
            checkpoint['global_params'] = model.get_global_params()
            checkpoint['local_params'] = model.get_local_params()
        
        checkpoint_path = f"{self.save_folder_name}/checkpoint_client_{self.id}_round_{self.current_round}{path_suffix}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path