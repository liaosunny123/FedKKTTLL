import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from flcore.trainmodel.models_fedext import FedEXTModel
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import wandb


class clientFedEXT(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Learning rate settings
        self.learning_rate_decay = getattr(args, 'learning_rate_decay', False)
        self.learning_rate_decay_gamma = getattr(args, 'learning_rate_decay_gamma', 0.99)
        self.current_round = 0

        # FedEXT specific attributes
        self.group_id = None  # Will be set based on distribution config
        self.feature_dim = args.feature_dim
        self.num_classes = args.num_classes

        # Contrastive learning parameters
        self.contrastive_weight = getattr(args, 'contrastive_weight', 0.1)  # Default to 0.1
        self.temperature = getattr(args, 'contrastive_temperature', 0.07)  # Standard SupCon temperature

        # Dynamic learning rate for faster convergence
        self.initial_lr = self.learning_rate
        self.warmup_rounds = 5  # Warm up for first 5 rounds

    def set_group(self, group_id):
        """Set the group ID for this client"""
        self.group_id = group_id
        print(f"Client {self.id} assigned to group {self.group_id}")

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)

        # Update current round
        self.current_round += 1

        # Simple learning rate strategy (same as FedAvg)
        current_lr = self.learning_rate
        if self.learning_rate_decay:
            current_lr = self.learning_rate * (self.learning_rate_decay_gamma ** self.current_round)

        # Use same optimizer settings as FedAvg for fair comparison
        encoder_optimizer = torch.optim.SGD(model.encoder.parameters(), lr=current_lr)
        classifier_optimizer = torch.optim.SGD(model.classifier.parameters(), lr=current_lr)

        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        epoch_losses = []
        for epoch in range(max_local_epochs):
            batch_losses = []
            epoch_classification_losses = []
            epoch_contrastive_losses = []
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
                features = model.encoder(x)

                # Classification loss
                classification_loss = self.loss(output, y)

                # Only add contrastive loss if weight > 0
                if self.contrastive_weight > 0:
                    contrastive_loss = self.compute_contrastive_loss(features, y)
                    total_loss = classification_loss + self.contrastive_weight * contrastive_loss
                else:
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                    total_loss = classification_loss

                batch_losses.append(total_loss.item())
                epoch_classification_losses.append(classification_loss.item())
                epoch_contrastive_losses.append(contrastive_loss.item())

                # Zero gradients for both optimizers
                encoder_optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                total_loss.backward()

                # Gradient clipping (optional - can be disabled for simpler training)
                # torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 10)
                # torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), 10)

                # Update both encoder and classifier
                encoder_optimizer.step()
                classifier_optimizer.step()

            # Calculate average loss for this epoch
            if batch_losses:
                epoch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(epoch_loss)

                # Log epoch metrics to wandb
                if self.use_wandb:
                    global_step = self.current_round * 1000 + epoch
                    avg_classification_loss = sum(epoch_classification_losses) / len(epoch_classification_losses) if epoch_classification_losses else 0
                    avg_contrastive_loss = sum(epoch_contrastive_losses) / len(epoch_contrastive_losses) if epoch_contrastive_losses else 0
                    wandb.log({
                        f"Client_{self.id}/epoch_loss": epoch_loss,
                        f"Client_{self.id}/classification_loss": avg_classification_loss,
                        f"Client_{self.id}/contrastive_loss": avg_contrastive_loss,
                        f"Client_{self.id}/learning_rate": current_lr,
                        f"Client_{self.id}/round": self.current_round,
                        f"Client_{self.id}/local_epoch": epoch,
                        f"Client_{self.id}/group_id": self.group_id,
                    }, step=global_step)

        save_item(model, self.role, 'model', self.save_folder_name)

        train_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += train_time

        # Log training summary to wandb
        if self.use_wandb:
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            global_step = self.current_round * 1000 + max_local_epochs
            wandb.log({
                f"Client_{self.id}/train_time": train_time,
                f"Client_{self.id}/total_epochs": max_local_epochs,
                f"Client_{self.id}/avg_train_loss": avg_loss,
                f"Client_{self.id}/global_round": self.current_round,
                f"Client_{self.id}/num_train_samples": len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0
            }, step=global_step)

        # Print training information
        if hasattr(trainloader.dataset, 'indices'):
            print(f"Client {self.id} (Group {self.group_id}): Trained on {len(trainloader.dataset.indices)} samples")
        else:
            print(f"Client {self.id} (Group {self.group_id}): Training completed")

    def compute_contrastive_loss(self, features, labels):
        """
        Supervised contrastive loss for better feature learning.
        Uses temperature-scaled cross-entropy to create better feature representations.

        This implementation is specifically designed for FedEXT to:
        1. Learn discriminative features within each client
        2. Create features that are more robust to aggregation
        3. Work well even with non-IID data distributions

        Args:
            features: [batch_size, feature_dim] - encoder output features
            labels: [batch_size] - class labels

        Returns:
            contrastive_loss: scalar tensor
        """
        batch_size = features.shape[0]

        # Skip if too small or contrastive weight is 0
        if batch_size < 2 or self.contrastive_weight == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # L2 normalize features for cosine similarity
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Apply temperature scaling
        similarity_matrix = similarity_matrix / self.temperature

        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask_positive = torch.eq(labels, labels.T).float().to(features.device)

        # Remove diagonal (self-similarity)
        diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        mask_positive = mask_positive.masked_fill(diagonal_mask, 0)

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Compute log probabilities
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits.masked_fill(diagonal_mask, 0)

        # Sum of exponentials for normalization (excluding positives for each anchor)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # Compute mean of log-likelihood over positive pairs
        # Only consider samples that have at least one positive pair
        mask_positive_sum = mask_positive.sum(dim=1)
        valid_samples = mask_positive_sum > 0

        if valid_samples.sum() == 0:
            # No valid positive pairs in batch
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        # Supervised contrastive loss
        loss = -(mask_positive * log_prob).sum(dim=1)
        loss = loss[valid_samples] / mask_positive_sum[valid_samples]
        loss = loss.mean()

        return loss

    def extract_features_labels(self):
        """
        Extract features and labels from local training data
        This is called in the last round to collect data for server training
        """
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)

        # Extract features from training data
        features, labels = model.extract_features(trainloader, self.device)

        print(f"Client {self.id}: Extracted {features.shape[0]} features with dimension {features.shape[1]}")

        return features, labels

    def receive_encoder(self, encoder_params):
        """Receive and update encoder parameters from server"""
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)  # Ensure model is on correct device

        # Only update encoder if we have valid parameters
        if encoder_params and len(encoder_params) > 0:
            # Move parameters to correct device
            device_params = {k: v.to(self.device) if hasattr(v, 'to') else v
                           for k, v in encoder_params.items()}
            model.set_encoder_params(device_params)

        save_item(model, self.role, 'model', self.save_folder_name)

    def receive_classifier(self, classifier_params):
        """Receive and update classifier parameters from group aggregation"""
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)  # Ensure model is on correct device

        # Only update classifier if we have valid parameters
        if classifier_params and len(classifier_params) > 0:
            # Move parameters to correct device
            device_params = {k: v.to(self.device) if hasattr(v, 'to') else v
                           for k, v in classifier_params.items()}
            model.set_classifier_params(device_params)

        save_item(model, self.role, 'model', self.save_folder_name)

    def get_encoder_params(self):
        """Get encoder parameters for aggregation"""
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)  # Ensure model is on correct device
        return model.get_encoder_params()

    def get_classifier_params(self):
        """Get classifier parameters for group aggregation"""
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)  # Ensure model is on correct device
        return model.get_classifier_params()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)  # CRITICAL: Move model to correct device!
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = model(x)

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

        # Log test metrics to wandb
        if self.use_wandb:
            wandb.log({
                f"Client_{self.id}/test_accuracy": test_acc / test_num if test_num > 0 else 0,
                f"Client_{self.id}/test_auc": auc,
                f"Client_{self.id}/test_samples": test_num
            })

        return test_acc, test_num, auc