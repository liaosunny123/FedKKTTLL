import copy
import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.data_distribution import DataDistributionManager
from flcore.trainmodel.models import BaseHeadSplit
import wandb


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.role = 'Client_' + str(self.id)
        self.save_folder_name = args.save_folder_name_full

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            model = BaseHeadSplit(args, self.id).to(self.device)
            save_item(model, self.role, 'model', self.save_folder_name)

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()

        # Use wandb from server (don't initialize here)
        self.use_wandb = getattr(args, 'use_wandb', False)

        # Initialize data distribution manager
        self.distribution_config = getattr(args, 'distribution_config', None)
        self.distribution_manager = DataDistributionManager(self.distribution_config)


    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        # Cache train data to avoid re-filtering
        if not hasattr(self, '_cached_train_data'):
            train_data = read_client_data(self.dataset, self.id, is_train=True)

            # Apply distribution configuration if available
            if self.distribution_manager and self.distribution_manager.config:
                train_data = self.distribution_manager.filter_client_data(
                    self.id, train_data, self.num_classes, is_train=True
                )
            self._cached_train_data = train_data

        return DataLoader(self._cached_train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        # Cache test data to avoid re-filtering
        if not hasattr(self, '_cached_test_data'):
            test_data = read_client_data(self.dataset, self.id, is_train=False)

            # Apply same distribution configuration to test data
            # This ensures domain-specific evaluation
            if self.distribution_manager and self.distribution_manager.config:
                test_data = self.distribution_manager.filter_client_data(
                    self.id, test_data, self.num_classes, is_train=False
                )
            self._cached_test_data = test_data

        return DataLoader(self._cached_test_data, batch_size, drop_last=False, shuffle=False)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
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

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        # Log metrics to wandb
        if self.use_wandb:
            wandb.log({
                f"Client_{self.id}/test_accuracy": test_acc / test_num,
                f"Client_{self.id}/test_auc": auc,
                f"Client_{self.id}/test_samples": test_num
            })

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # Log training metrics to wandb
        if self.use_wandb:
            wandb.log({
                f"Client_{self.id}/train_loss": losses / train_num if train_num > 0 else 0,
                f"Client_{self.id}/train_samples": train_num
            })

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


def save_item(item, role, item_name, item_path=None):
    if not os.path.exists(item_path):
        os.makedirs(item_path)
    torch.save(item, os.path.join(item_path, role + "_" + item_name + ".pt"))

def load_item(role, item_name, item_path=None):
    try:
        return torch.load(os.path.join(item_path, role + "_" + item_name + ".pt"), 
                  weights_only=False)
    except FileNotFoundError:
        print(role, item_name, 'Not Found')
        return None
