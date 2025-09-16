import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.utils.data import DataLoader
import wandb


class clientKTL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu
        self.ETF_dim = args.num_classes

        self.m = 0.5
        self.s = 64

        self.classes_ids_tensor = torch.tensor(list(range(self.num_classes)),
                                               dtype=torch.int64, device=self.device)
        self.MSEloss = nn.MSELoss()

        # 添加学习率衰减支持
        self.learning_rate_decay = getattr(args, 'learning_rate_decay', False)
        self.learning_rate_decay_gamma = getattr(args, 'learning_rate_decay_gamma', 0.99)
        self.current_round = 0
        self.train_step_count = 0  # 用于跟踪训练步数


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        ETF = load_item('Server', 'ETF', self.save_folder_name)
        ETF = F.normalize(ETF.T)

        data_generated = load_item('Server', 'data_generated', self.save_folder_name)
        if data_generated is not None:
            gen_loader = DataLoader(data_generated, self.batch_size, drop_last=False, shuffle=True)
            gen_iter = iter(gen_loader)
        else:
            print(f"Client {self.id}: No generated data available")
        proj_fc = load_item('Server', 'proj_fc', self.save_folder_name)

        # 更新当前轮次
        self.current_round += 1

        # 应用学习率衰减
        current_lr = self.learning_rate
        if self.learning_rate_decay:
            current_lr = self.learning_rate * (self.learning_rate_decay_gamma ** self.current_round)

        optimizer = torch.optim.SGD(model.parameters(), lr=current_lr)
        opt_proj_fc = torch.optim.SGD(proj_fc.parameters(), lr=current_lr)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        epoch_losses = []
        # Debug: Check data distribution for Client 0
        if self.id == 0 and self.current_round == 1:
            label_counts = defaultdict(int)
            for _, y in trainloader:
                if isinstance(y, torch.Tensor):
                    for label in y.cpu().numpy():
                        label_counts[label] += 1
            print(f"Client {self.id} label distribution: {dict(label_counts)}")
            print(f"Client {self.id} total samples: {sum(label_counts.values())}")

        for step in range(max_local_epochs):
            batch_losses = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                proj = model(x)
                proj = F.normalize(proj)
                cosine = F.linear(proj, ETF)

                # ArcFace loss with clipping to prevent NaN
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                # Clip cosine values to prevent NaN in arccos
                cosine_clipped = torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7)
                arccos = torch.acos(cosine_clipped)
                cosine_new = torch.cos(arccos + self.m)
                cosine = one_hot * cosine_new + (1 - one_hot) * cosine
                cosine = cosine * self.s
                loss = self.loss(cosine, y)
                batch_losses.append(loss.item())

                # knowledge transfer
                if data_generated is not None:
                    try:
                        (x_G, y_G) = next(gen_iter)
                    except StopIteration:
                        gen_iter = iter(gen_loader)
                        (x_G, y_G) = next(gen_iter)
            
                    if type(x_G) == type([]):
                        x_G[0] = x_G[0].to(self.device)
                    else:
                        x_G = x_G.to(self.device)
                    y_G = y_G.to(self.device)

                    rep_G = model.base(x_G)
                    proj_G = proj_fc(rep_G)
                    
                    loss += self.MSEloss(proj_G, y_G) * self.mu

                optimizer.zero_grad()
                opt_proj_fc.zero_grad()
                loss.backward()

                # Debug gradient norms
                if self.id == 0 and i == 0 and step == 0:
                    total_norm = 0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2.)
                    print(f"Client {self.id} gradient norm before clipping: {total_norm:.4f}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                torch.nn.utils.clip_grad_norm_(proj_fc.parameters(), 100)
                optimizer.step()
                opt_proj_fc.step()

            # Calculate average loss for this epoch
            if batch_losses:
                epoch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(epoch_loss)

                # Log epoch metrics to wandb
                if self.use_wandb:
                    self.train_step_count += 1
                    global_step = (self.current_round - 1) * max_local_epochs + step
                    wandb.log({
                        f"Client_{self.id}/step": global_step,
                        f"Client_{self.id}/epoch_loss": epoch_loss,
                        f"Client_{self.id}/learning_rate": current_lr,
                        f"Client_{self.id}/round": self.current_round,
                        f"Client_{self.id}/local_epoch": step,
                    }, step=global_step)

        save_item(model, self.role, 'model', self.save_folder_name)

        train_time = time.time() - start_time
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += train_time

        # Log training summary to wandb
        if self.use_wandb:
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            global_step = self.current_round * max_local_epochs
            wandb.log({
                f"Client_{self.id}/train_time": train_time,
                f"Client_{self.id}/total_epochs": max_local_epochs,
                f"Client_{self.id}/avg_train_loss": avg_loss,
                f"Client_{self.id}/global_round": self.current_round,
                f"Client_{self.id}/num_train_samples": len(trainloader.dataset) if hasattr(trainloader, 'dataset') else 0
            }, step=global_step)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        ETF = load_item('Server', 'ETF', self.save_folder_name)
        ETF = F.normalize(ETF.T)
        # model.to(self.device)
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
                proj = model(x)
                cosine = F.linear(F.normalize(proj), ETF)

                test_acc += (torch.sum(torch.argmax(cosine, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(cosine.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        # Log test metrics to wandb
        if self.use_wandb:
            wandb.log({
                f"Client_{self.id}/test_accuracy": test_acc / test_num if test_num > 0 else 0,
                f"Client_{self.id}/test_auc": auc,
                f"Client_{self.id}/test_samples": test_num
            })

        return test_acc, test_num, auc

    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)

        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                proj = model(x)
                proj = F.normalize(proj)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(proj[i, :].detach().data)

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)


def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = F.normalize(proto / len(proto_list), dim=0)
        else:
            protos[label] = F.normalize(proto_list[0], dim=0)

    return protos