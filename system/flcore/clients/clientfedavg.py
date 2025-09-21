import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import wandb


class clientFedAvg(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # Learning rate settings
        self.learning_rate_decay = getattr(args, 'learning_rate_decay', False)
        self.learning_rate_decay_gamma = getattr(args, 'learning_rate_decay_gamma', 0.99)
        self.current_round = 0

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)

        # Update current round
        self.current_round += 1

        # Apply learning rate decay
        current_lr = self.learning_rate
        if self.learning_rate_decay:
            current_lr = self.learning_rate * (self.learning_rate_decay_gamma ** self.current_round)

        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr)
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

                output = model(x)
                loss = self.loss(output, y)
                batch_losses.append(loss.item())
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

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

                # Log epoch metrics to wandb with client-specific step
                if self.use_wandb:
                    # 使用统一的全局step计算
                    global_step = (self.current_round - 1) * max_local_epochs + epoch
                    
                    # 每个客户端记录自己的step
                    log_dict = {
                        f"Client_{self.id}/step": global_step,
                        f"Client_{self.id}/epoch_loss": epoch_loss,
                        f"Client_{self.id}/epoch_accuracy": epoch_acc,
                        f"Client_{self.id}/learning_rate": current_lr,
                        f"Client_{self.id}/round": self.current_round,
                        f"Client_{self.id}/local_epoch": epoch,
                    }
                    
                    # 为每个客户端定义自己的x轴
                    wandb.define_metric(f"Client_{self.id}/step")
                    wandb.define_metric(f"Client_{self.id}/*", step_metric=f"Client_{self.id}/step")
                    
                    wandb.log(log_dict)

        save_item(model, self.role, 'model', self.save_folder_name)

        train_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += train_time

        # Calculate training summary
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        avg_acc = sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0

        # Log training summary to wandb
        if self.use_wandb:
            # 使用轮次结束时的step
            global_step = self.current_round * max_local_epochs
            
            log_dict = {
                f"Client_{self.id}/step": global_step,
                f"Client_{self.id}/train_time": train_time,
                f"Client_{self.id}/total_epochs": max_local_epochs,
                f"Client_{self.id}/avg_train_loss": avg_loss,
                f"Client_{self.id}/avg_train_accuracy": avg_acc,
                f"Client_{self.id}/global_round": self.current_round,
                f"Client_{self.id}/num_train_samples": len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0
            }
            
            wandb.log(log_dict)

        # Print training information with details
        num_samples = len(trainloader.dataset.indices) if hasattr(trainloader.dataset, 'indices') else len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 'unknown'
        print(f"Client {self.id}: "
              f"Round {self.current_round} - "
              f"Trained on {num_samples} samples, "
              f"Avg loss: {avg_loss:.4f}, "
              f"Avg acc: {avg_acc:.4f}")

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
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
        
        test_accuracy = test_acc / test_num if test_num > 0 else 0

        # Log test metrics to wandb with client-specific step
        if self.use_wandb:
            # 测试时的step对齐到训练结束
            global_step = self.current_round * self.local_epochs
            
            log_dict = {
                f"Client_{self.id}/step": global_step,
                f"Client_{self.id}/test_accuracy": test_accuracy,
                f"Client_{self.id}/test_auc": auc,
                f"Client_{self.id}/test_samples": test_num
            }
            
            # 确保test metrics使用相同的step定义
            wandb.define_metric(f"Client_{self.id}/step")
            wandb.define_metric(f"Client_{self.id}/test_*", step_metric=f"Client_{self.id}/step")
            
            wandb.log(log_dict)

        print(f"Client {self.id} Test Results: "
              f"Acc={test_accuracy:.4f}, AUC={auc:.4f}, Samples={test_num}")

        return test_acc, test_num, auc