import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import shutil
from utils.data_utils import read_client_data
from flcore.clients.clientbase import load_item, save_item
import wandb


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.role = 'Server'
        if args.save_folder_name == 'temp':
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/{time.time()}/'
        elif 'temp' in args.save_folder_name:
            args.save_folder_name_full = args.save_folder_name
        else:
            args.save_folder_name_full = f'{args.save_folder_name}/{args.dataset}/{args.algorithm}/'
        self.save_folder_name = args.save_folder_name_full

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        # Initialize wandb for server and all clients
        self.use_wandb = getattr(args, 'use_wandb', False)
        if self.use_wandb:
            run_name = f"{self.algorithm}_{self.dataset}_clients{self.num_clients}_run{times}"
            wandb.init(
                project="fedktl",
                entity="epicmo",
                name=run_name,
                config={
                    "algorithm": self.algorithm,
                    "dataset": self.dataset,
                    "num_clients": self.num_clients,
                    "num_classes": self.num_classes,
                    "global_rounds": self.global_rounds,
                    "local_epochs": self.local_epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "join_ratio": self.join_ratio,
                    "eval_gap": self.eval_gap,
                    "client_drop_rate": self.client_drop_rate,
                    "train_slow_rate": self.train_slow_rate,
                    "send_slow_rate": self.send_slow_rate
                },
                reinit=True
            )
            # Define custom x-axis for metrics
            wandb.define_metric("Global/round")
            wandb.define_metric("Global/*", step_metric="Global/round")

            # Define step metric for each client
            for i in range(self.num_clients):
                wandb.define_metric(f"Client_{i}/step")
                wandb.define_metric(f"Client_{i}/epoch_loss", step_metric=f"Client_{i}/step")
                wandb.define_metric(f"Client_{i}/learning_rate", step_metric=f"Client_{i}/step")

                # Other client metrics use global round as step
                wandb.define_metric(f"Client_{i}/test_accuracy", step_metric="Global/round")
                wandb.define_metric(f"Client_{i}/test_auc", step_metric="Global/round")
                wandb.define_metric(f"Client_{i}/train_time", step_metric="Global/round")
                wandb.define_metric(f"Client_{i}/avg_train_loss", step_metric="Global/round")


    def set_clients(self, clientObj):
        print("\n" + "="*60)
        print("Initializing client data distributions...")
        print("="*60)

        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=train_slow,
                            send_slow=send_slow)
            self.clients.append(client)

        print("="*60)
        print("Client data distributions initialized.")
        print("="*60 + "\n")

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        client = self.clients[self.uploaded_ids[0]]
        global_model = load_item(client.role, 'model', client.save_folder_name)
        for param in global_model.parameters():
            param.data.zero_()
            
        for w, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            for server_param, client_param in zip(global_model.parameters(), client_model.parameters()):
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        
    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
        
        if 'temp' in self.save_folder_name:
            try:
                shutil.rmtree(self.save_folder_name)
                print('Deleted.')
            except:
                print('Already deleted.')

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct*1.0)
            acc = ct*1.0/ns
            print(f'Client {c.id}: Acc: {acc}, AUC: {auc}')
            tot_auc.append(auc*ns)
            num_samples.append(ns)

            # Individual client metrics are logged by clients themselves
            # No need to log here to avoid duplication

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        # stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        # Log global metrics to wandb
        if self.use_wandb:
            current_round = len(self.rs_test_acc)
            wandb.log({
                "Global/round": current_round,
                "Global/test_accuracy": test_acc,
                "Global/test_auc": test_auc,
                "Global/num_selected_clients": self.current_num_join_clients,
                "Global/mean_client_accuracy": np.mean(accs),
                "Global/std_client_accuracy": np.std(accs),
                "Global/mean_client_auc": np.mean(aucs),
                "Global/std_client_auc": np.std(aucs)
            })

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        # if loss == None:
        #     self.rs_train_loss.append(train_loss)
        # else:
        #     loss.append(train_loss)

        # print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True
