import time
import numpy as np
import torch
import torch.nn as nn
import copy
from flcore.servers.serverbase import Server
from flcore.clients.clientfedavg import clientFedAvg
from flcore.clients.clientbase import load_item, save_item
from flcore.trainmodel.models import BaseHeadSplit
from threading import Thread
import wandb


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFedAvg)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedAvg server and clients.")

        self.Budget = []
        self.args = args  # Store args for later use
        self.use_global_model = args.use_global_model and args.is_homogeneity_model

        # Initialize models
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            # Initialize client models first
            for client in self.clients:
                model = BaseHeadSplit(args, client.id).to(self.device)
                save_item(model, client.role, 'model', client.save_folder_name)

            # Initialize global model for homogeneous setting
            if self.use_global_model:
                print("Initializing global model for FedAvg...")
                # Use the same model structure as clients
                self.global_model = BaseHeadSplit(args, 0).to(self.device)
                save_item(self.global_model, self.role, 'global_model', self.save_folder_name)
                print('Global model initialized for FedAvg')

    def train(self):
        # Initial evaluation
        print(f"\n-------------Initial Evaluation-------------")
        print(f"USE_GLOBAL_MODEL: {self.use_global_model}")
        print(f"IS_HOMOGENEITY_MODEL: {self.args.is_homogeneity_model}")
        print("\nEvaluate initial models performance")
        self.evaluate()

        if self.use_global_model:
            print("\nEvaluate initial global model performance")
            self.test_global_model()
        else:
            print(f"\nGlobal model evaluation skipped (use_global_model={self.use_global_model})")

        print("\nStarting FedAvg training...")

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
            if self.use_global_model and len(self.selected_clients) > 0:
                self.aggregate_parameters()

            # Evaluation
            if i % self.eval_gap == 0:
                print("\nEvaluate models after aggregation")
                self.evaluate()

                if self.use_global_model:
                    print("\nEvaluate global model performance")
                    self.test_global_model()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, 'Round time cost:', self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy:")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round:")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]) if len(self.Budget) > 1 else 0)

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
        """Send global model to selected clients"""
        assert (len(self.selected_clients) > 0)

        if self.use_global_model:
            # Send global model to all selected clients
            global_model = load_item(self.role, 'global_model', self.save_folder_name)
            for client in self.selected_clients:
                save_item(copy.deepcopy(global_model), client.role, 'model', client.save_folder_name)

    def receive_models(self):
        """Receive models from selected clients"""
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0

        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)

            if self.use_global_model:
                # Load client model for aggregation
                client_model = load_item(client.role, 'model', client.save_folder_name)
                self.uploaded_models.append(client_model)

        # Normalize weights
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        """Aggregate client models using weighted average"""
        assert (len(self.uploaded_models) > 0)

        # Load global model
        global_model = load_item(self.role, 'global_model', self.save_folder_name)

        # Get global model parameters
        global_params = global_model.state_dict()

        # Initialize aggregated parameters
        for key in global_params.keys():
            global_params[key] = torch.zeros_like(global_params[key])

        # Weighted average of client models
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            client_params = client_model.state_dict()
            for key in global_params.keys():
                if key in client_params:
                    global_params[key] += client_params[key] * w

        # Update global model
        global_model.load_state_dict(global_params)
        save_item(global_model, self.role, 'global_model', self.save_folder_name)

        print(f"Aggregated {len(self.uploaded_models)} client models")

    def test_global_model(self):
        """Test global model on aggregated test dataset"""
        if not self.use_global_model:
            return None, None, None

        # Load global model
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        global_model.eval()

        # Aggregate test results from all clients
        test_acc = 0
        test_num = 0

        for client in self.clients:
            testloader = client.load_test_data()

            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output = global_model(x)
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

        accuracy = test_acc / test_num if test_num > 0 else 0

        print(f"Server Global Model: Acc: {accuracy:.4f}, Samples: {test_num}")

        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "Server/test_accuracy": accuracy,
                "Server/test_samples": test_num,
                "Server/round": len(self.rs_test_acc)
            })

        return test_acc, test_num, accuracy