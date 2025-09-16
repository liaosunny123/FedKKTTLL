import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from flcore.clients.clientktl import clientKTL
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from flcore.trainmodel.models import BaseHeadSplit
from threading import Thread
from collections import defaultdict
from torch.utils.data import DataLoader
import wandb


class FedKTL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientKTL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.feature_dim = args.feature_dim
        self.server_learning_rate = args.server_learning_rate
        self.gen_batch_size = args.gen_batch_size
        self.server_batch_size = args.server_batch_size
        self.server_epochs = args.server_epochs
        self.lamda = args.lamda
        self.use_etf = args.use_etf
        self.use_global_model = args.use_global_model and args.is_homogeneity_model
        self.use_prototype_aggregation = args.use_prototype_aggregation
        self.ETF_dim = args.num_classes
        self.classes_ids_tensor = torch.tensor(list(range(self.num_classes)),
                                               dtype=torch.int64, device=self.device)

        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            trainloader = self.clients[0].load_train_data()
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                self.img_shape = x[0].shape
                break

            # Check if using prototype aggregation mode
            if self.use_prototype_aggregation:
                print("\n===== Using Prototype Aggregation Mode (FedProto-like) =====")
                print("Bypassing generator initialization...")

                # Use feature_dim as prototype dimension
                self.prototype_dim = self.feature_dim

                # Initialize projection layer for clients
                proj_fc = nn.Linear(self.feature_dim, self.prototype_dim).to(self.device)
                save_item(proj_fc, self.role, 'proj_fc', self.save_folder_name)
                print(f"Projection layer: {self.feature_dim} -> {self.prototype_dim}")

                # Initialize global prototypes storage
                self.global_prototypes = {}
                for c in range(self.num_classes):
                    self.global_prototypes[c] = torch.zeros(self.prototype_dim).to(self.device)
                save_item(self.global_prototypes, self.role, 'global_prototypes', self.save_folder_name)
                print(f"Initialized {self.num_classes} global prototypes with dim={self.prototype_dim}")

                # Initialize ETF if needed
                if self.use_etf:
                    while True:
                        try:
                            P = generate_random_orthogonal_matrix(self.ETF_dim, self.num_classes)
                            I = torch.eye(self.num_classes)
                            one = torch.ones(self.num_classes, self.num_classes)
                            F_mat = np.sqrt(self.num_classes / (self.num_classes-1)) * torch.matmul(P, I-((1/self.num_classes) * one))
                            ETF = F_mat.requires_grad_(False).to(self.device)
                            save_item(ETF, self.role, 'ETF', self.save_folder_name)
                            print('ETF initialized for prototype mode')
                            break
                        except AssertionError:
                            pass

                # Initialize global model if enabled
                if self.use_global_model:
                    print("Initializing global model for prototype-based training...")
                    if self.use_etf:
                        # When using ETF, create a model that maps prototypes to ETF space
                        self.global_model = nn.Sequential(
                            nn.Linear(self.prototype_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, self.ETF_dim)
                        ).to(self.device)
                    else:
                        # Normal classifier for prototype classification
                        self.global_model = nn.Sequential(
                            nn.Linear(self.prototype_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, self.num_classes)
                        ).to(self.device)
                    save_item(self.global_model, self.role, 'global_model', self.save_folder_name)
                    print('Global model (prototype-based) initialized')

            else:
                # Original generator-based initialization
                print("\n===== Using Generator Mode (Original FedKTL) =====")
                from diffusers import StableDiffusionPipeline
                import os

                # Always try to load from Hugging Face Hub (with local caching)
                print(f"Loading Stable Diffusion model: {args.generator_path}")
                try:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        args.generator_path,
                        dtype=torch.float32,  # Use 'dtype' instead of deprecated 'torch_dtype'
                        use_safetensors=True
                    )
                    print("✅ Successfully loaded model from Hugging Face Hub")
                except Exception as e:
                    print(f"Failed to load from Hub: {e}")
                    print("Trying to load from local cache...")
                    pipe = StableDiffusionPipeline.from_pretrained(
                        args.generator_path,
                        dtype=torch.float32,
                        local_files_only=True,
                        use_safetensors=True
                    )
                pipe.set_progress_bar_config(disable=True)
                pipe = pipe.to(self.device)

                self.pipe_num_inference_steps = 50
                self.pipe_num_images_per_prompt = 1 # set 1 to reduce GPU memory
                self.pipe_height = pipe.unet.config.sample_size * pipe.vae_scale_factor
                self.pipe_width = pipe.unet.config.sample_size * pipe.vae_scale_factor
                self.pipe_guidance_scale = 7.5
                self.pipe_negative_prompt = None
                self.pipe_eta = 0.0
                self.pipe_generator = None
                self.pipe_latents = None
                self.pipe_prompt_embeds = None
                self.pipe_negative_prompt_embeds = None
                self.pipe_output_type = "pil"
                self.pipe_return_dict = True
                self.pipe_callback = None
                self.pipe_callback_steps = 1
                self.pipe_cross_attention_kwargs = None
                self.pipe_guidance_rescale = 0.0
                self.pipe_device = pipe._execution_device
                self.pipe_do_classifier_free_guidance = self.pipe_guidance_scale > 1.0
                self.pipe_text_encoder_lora_scale = (
                    self.pipe_cross_attention_kwargs.get("scale", None) if self.pipe_cross_attention_kwargs is not None else None
                )
                self.pipe_num_channels_latents = pipe.unet.config.in_channels
                self.pipe_prompt = args.stable_diffusion_prompt

                self.pipe_prompt_embeds, self.pipe_negative_prompt_embeds = pipe.encode_prompt(
                    self.pipe_prompt,
                    self.pipe_device,
                    self.pipe_num_images_per_prompt,
                    self.pipe_do_classifier_free_guidance,
                    self.pipe_negative_prompt,
                    prompt_embeds=self.pipe_prompt_embeds,
                    negative_prompt_embeds=self.pipe_negative_prompt_embeds,
                    lora_scale=self.pipe_text_encoder_lora_scale,
                )

                latents = pipe.prepare_latents(
                            self.pipe_num_images_per_prompt,
                            self.pipe_num_channels_latents,
                            self.pipe_height,
                            self.pipe_width,
                            self.pipe_prompt_embeds.dtype,
                            self.pipe_device,
                            self.pipe_generator,
                            None,
                        )
                print('latents shape', latents.shape)
                self.pipe_latent_shape = latents.shape
                pipe.latent_dim = latents.view(-1).shape[0]
                print('latents dim', pipe.latent_dim)
                save_item(pipe, self.role, 'pipe', self.save_folder_name)

                # Initialize Feature_Transformer
                if self.use_etf:
                    F = Feature_Transformer(self.ETF_dim, pipe.latent_dim).to(self.device)
                    save_item(F, self.role, 'F', self.save_folder_name)
                    print('Feature_Transformer', F)
                else:
                    F = Feature_Transformer(self.num_classes, pipe.latent_dim).to(self.device)
                    save_item(F, self.role, 'F', self.save_folder_name)
                    print('Feature_Transformer (non-ETF)', F)

                Centroids = nn.Embedding(self.num_classes, pipe.latent_dim).to(self.device)
                save_item(Centroids, self.role, 'Centroids', self.save_folder_name)
                print('Centroids', Centroids)

                # Initialize ETF only if use_etf is enabled
                if self.use_etf:
                    while True:
                        try:
                            P = generate_random_orthogonal_matrix(self.ETF_dim, self.num_classes)
                            I = torch.eye(self.num_classes)
                            one = torch.ones(self.num_classes, self.num_classes)
                            F = np.sqrt(self.num_classes / (self.num_classes-1)) * torch.matmul(P, I-((1/self.num_classes) * one))
                            ETF = F.requires_grad_(False).to(self.device)
                            save_item(ETF, self.role, 'ETF', self.save_folder_name)
                            print('ETF initialized')
                            break
                        except AssertionError:
                            pass
                else:
                    print('ETF classifier disabled')

                clientprocess = transforms.Compose(
                    [transforms.Resize(size=self.img_shape[-1]),
                    transforms.CenterCrop(size=(self.img_shape[-1], self.img_shape[-1])),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                save_item(clientprocess, self.role, 'clientprocess', self.save_folder_name)
                print('clientprocess', clientprocess)

                proj_fc = nn.Linear(self.feature_dim, pipe.latent_dim).to(self.device)
                save_item(proj_fc, self.role, 'proj_fc', self.save_folder_name)

                # Initialize global model if enabled and homogeneous
                if self.use_global_model:
                    print("Initializing global model for homogeneous setting...")
                    # Create a simple MLP classifier for prototype-based training
                    # Input: num_classes (prototype dimension in this system)
                    # Output: num_classes for classification
                    if self.use_etf:
                        # When using ETF, we only need to project to ETF_dim
                        self.global_model = nn.Sequential(
                            nn.Linear(self.num_classes, 64),
                            nn.ReLU(),
                            nn.Linear(64, self.ETF_dim)
                        ).to(self.device)
                    else:
                        # Normal classifier: identity or simple transform
                        self.global_model = nn.Sequential(
                            nn.Linear(self.num_classes, 64),
                            nn.ReLU(),
                            nn.Linear(64, self.num_classes)
                        ).to(self.device)
                    save_item(self.global_model, self.role, 'global_model', self.save_folder_name)
                    print('Global model (MLP classifier) initialized')

        self.MSEloss = nn.MSELoss()


    def train(self):
        # Create global test dataset if global model is enabled
        if self.use_global_model:
            self.create_global_test_dataset()

        # 先评估初始性能
        print(f"\n-------------Initial Evaluation-------------")
        print("\nEvaluate initial heterogeneous models performance")
        self.evaluate()

        # Test initial global model if enabled
        if self.use_global_model:
            print("\nEvaluate initial global model performance")
            self.test_global_model()

        print("\nStarting training...")

        for i in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            print(f"\n-------------Round number: {i}-------------")

            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()

            if self.use_prototype_aggregation:
                # Use prototype aggregation instead of alignment and generation
                self.aggregate_prototypes()
                self.send_prototypes()
            else:
                # Original FedKTL process
                self.align()
                self.generate_images(i)

            # Train and test global model if enabled
            if self.use_global_model:
                if self.use_prototype_aggregation:
                    self.train_global_model_proto(i)
                else:
                    self.train_global_model(i)
                self.test_global_model()

            # 训练后评估
            if i%self.eval_gap == 0:
                print("\nEvaluate heterogeneous models after training")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        # 训练结束后生成少量样本用于可视化
        if not self.use_prototype_aggregation:
            self.generate_final_samples()

    def aggregate_prototypes(self):
        """Aggregate prototypes at parameter level (FedProto-like)"""
        print("\n----Server aggregating prototypes (parameter-level)----")

        # Load uploaded prototypes
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
        global_prototypes = load_item(self.role, 'global_prototypes', self.save_folder_name)

        # Create a dictionary to store prototypes per class
        class_protos = defaultdict(list)
        class_weights = defaultdict(list)

        # Group prototypes by class
        for proto, label in uploaded_protos:
            class_label = label.item()
            class_protos[class_label].append(proto)
            class_weights[class_label].append(1.0)  # Equal weight for now

        # Average prototypes for each class
        for c in range(self.num_classes):
            if c in class_protos:
                # Stack and average prototypes
                stacked_protos = torch.stack(class_protos[c])
                weights = torch.tensor(class_weights[c]).to(self.device)
                weights = weights / weights.sum()

                # Weighted average
                global_prototypes[c] = (stacked_protos * weights.unsqueeze(1)).sum(dim=0)
            # If no prototype for this class, keep the previous one

        # Save aggregated prototypes
        save_item(global_prototypes, self.role, 'global_prototypes', self.save_folder_name)
        print(f"Aggregated prototypes for {len(class_protos)} classes")

    def send_prototypes(self):
        """Send global prototypes to clients"""
        print("\n----Sending global prototypes to clients----")
        global_prototypes = load_item(self.role, 'global_prototypes', self.save_folder_name)

        for client in self.selected_clients:
            # Save prototypes in client's folder
            save_item(global_prototypes, client.role, 'global_prototypes', client.save_folder_name)

    def train_global_model_proto(self, round_idx):
        """Train global model using aggregated prototypes"""
        if not self.use_global_model:
            return

        print(f"\n==== Training Global Model with Prototypes (Round {round_idx}) ====")

        # Load components
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        global_prototypes = load_item(self.role, 'global_prototypes', self.save_folder_name)

        if self.use_etf:
            ETF = load_item(self.role, 'ETF', self.save_folder_name)
            ETF = F.normalize(ETF.T)

        # Create training data from prototypes
        proto_data = []
        for c in range(self.num_classes):
            if torch.norm(global_prototypes[c]) > 0:  # Only use non-zero prototypes
                proto_data.append((global_prototypes[c], c))

        if len(proto_data) == 0:
            print("No valid prototypes to train global model")
            return

        proto_loader = DataLoader(proto_data, batch_size=min(len(proto_data), self.server_batch_size),
                                 shuffle=True)

        # Setup optimizer
        optimizer = torch.optim.SGD(global_model.parameters(),
                                   lr=self.server_learning_rate)
        global_model.train()

        # Training loop
        for epoch in range(min(20, self.server_epochs // 5)):  # Fewer epochs for prototype training
            epoch_loss = 0
            for protos, labels in proto_loader:
                protos = protos.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = global_model(protos)

                if self.use_etf:
                    # Compute loss with ETF classifier
                    logits = outputs @ ETF
                    loss = F.cross_entropy(logits, labels)
                else:
                    # Standard cross-entropy loss
                    loss = F.cross_entropy(outputs, labels)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss/len(proto_loader):.4f}")

        # Save updated global model
        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        print("Global model training with prototypes completed")

    def create_global_test_dataset(self):
        """Create a balanced test dataset for global model evaluation"""
        print("Creating global test dataset...")

        # Collect test data from all clients
        all_test_data = []
        all_test_labels = []

        for client in self.clients:
            test_loader = client.load_test_data()
            for x, y in test_loader:
                all_test_data.append(x)
                all_test_labels.append(y)

        # Concatenate all data
        if isinstance(all_test_data[0], list):
            # Handle list of tensors
            all_test_data = [torch.cat([d[i] for d in all_test_data])
                             for i in range(len(all_test_data[0]))]
        else:
            all_test_data = torch.cat(all_test_data)
        all_test_labels = torch.cat(all_test_labels)

        # Create balanced subset (100 samples per class if available)
        balanced_data = []
        balanced_labels = []
        samples_per_class = 100

        for c in range(self.num_classes):
            class_indices = (all_test_labels == c).nonzero(as_tuple=True)[0]
            if len(class_indices) > 0:
                # Sample up to samples_per_class
                num_samples = min(samples_per_class, len(class_indices))
                sampled_indices = class_indices[torch.randperm(len(class_indices))[:num_samples]]

                if isinstance(all_test_data, list):
                    class_data = [data[sampled_indices] for data in all_test_data]
                    balanced_data.extend(zip(*class_data))
                else:
                    balanced_data.append(all_test_data[sampled_indices])
                balanced_labels.append(all_test_labels[sampled_indices])

        if isinstance(all_test_data, list):
            # Already handled in the loop above
            balanced_labels = torch.cat(balanced_labels)
        else:
            balanced_data = torch.cat(balanced_data)
            balanced_labels = torch.cat(balanced_labels)

        # Create dataset
        from torch.utils.data import TensorDataset
        if isinstance(balanced_data, list):
            # Convert list of tuples back to proper format
            num_samples = len(balanced_data)
            num_inputs = len(balanced_data[0])
            reorganized_data = []
            for i in range(num_inputs):
                reorganized_data.append(torch.stack([balanced_data[j][i] for j in range(num_samples)]))
            self.global_test_dataset = TensorDataset(*reorganized_data, balanced_labels)
        else:
            self.global_test_dataset = TensorDataset(balanced_data, balanced_labels)

        print(f"Global test dataset created with {len(self.global_test_dataset)} samples")

    def test_global_model(self):
        """Test the global model performance"""
        if not self.use_global_model or not hasattr(self, 'global_test_dataset'):
            return

        print("Testing global model...")
        global_model = load_item(self.role, 'global_model', self.save_folder_name)

        if self.use_etf:
            ETF = load_item(self.role, 'ETF', self.save_folder_name)
            ETF = F.normalize(ETF.T)

        if self.use_prototype_aggregation:
            # For prototype mode, we need to extract features first
            global_prototypes = load_item(self.role, 'global_prototypes', self.save_folder_name)
            proj_fc = load_item(self.role, 'proj_fc', self.save_folder_name)

            # Create a simple prototype-based classifier
            def classify_with_prototypes(features):
                """Classify by finding nearest prototype"""
                distances = []
                for c in range(self.num_classes):
                    proto = global_prototypes[c].unsqueeze(0)
                    dist = torch.cdist(features, proto)
                    distances.append(dist)
                distances = torch.cat(distances, dim=1)
                return -distances  # Negative distance as logits

        test_loader = DataLoader(self.global_test_dataset, batch_size=64, shuffle=False)

        global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    data, labels = batch
                    data = data.to(self.device)
                else:
                    # Multiple inputs
                    data = batch[0].to(self.device)
                    labels = batch[-1]
                labels = labels.to(self.device)

                if self.use_prototype_aggregation:
                    # Extract features and classify with prototypes
                    # For simplicity, using the first model's feature extractor
                    client_model = self.clients[0].model
                    if hasattr(client_model, 'base'):
                        features = client_model.base(data)
                    else:
                        # Assume the model has a feature extraction part
                        features = data  # Placeholder

                    features = proj_fc(features)
                    logits = classify_with_prototypes(features)
                else:
                    # Use global model directly
                    outputs = global_model(data)
                    if self.use_etf:
                        logits = outputs @ ETF
                    else:
                        logits = outputs

                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Global Model Test Accuracy: {accuracy:.2f}%")

        # Log to wandb if enabled
        if self.args.use_wandb:
            wandb.log({"global_model_test_accuracy": accuracy})

        return accuracy

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        uploaded_protos = []
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)

            protos = load_item(client.role, 'protos', client.save_folder_name)
            for cc in protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                uploaded_protos.append((protos[cc], y))

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        save_item(uploaded_protos, self.role, 'uploaded_protos', self.save_folder_name)

    @torch.no_grad()
    def set_Centroids(self, uploaded_protos, F, Centroids): # set Centroids to the centroids of latent vectors
        proto_loader = DataLoader(uploaded_protos, self.server_batch_size, drop_last=False, shuffle=True)

        protos = defaultdict(list)
        F.eval()
        for P, y in proto_loader:
            Q = F(P).detach()
            for i, yy in enumerate(y):
                y_c = yy.item()
                protos[y_c].append(Q[i, :].data)

        protos = avg_func(protos)
        for i, weight in enumerate(Centroids.weight):
            if type(protos[i]) != type([]):
                weight.data = protos[i].data.clone()

    def align(self):
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
        pipe = load_item(self.role, 'pipe', self.save_folder_name)
        F = load_item(self.role, 'F', self.save_folder_name)
        Centroids = load_item(self.role, 'Centroids', self.save_folder_name)

        self.set_Centroids(uploaded_protos, F, Centroids)

        opt_F = torch.optim.Adam(F.parameters(),
                                 lr=self.server_learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-08,
                                 weight_decay=0,
                                 amsgrad=False)
        opt_Centroids = torch.optim.Adam(Centroids.parameters(),
                                    lr=self.server_learning_rate,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=0,
                                    amsgrad=False)

        print('\n----Server aligning ...----\n')

        F.train()
        Centroids.train()
        for _ in range(self.server_epochs):
            proto_loader = DataLoader(uploaded_protos, self.server_batch_size, drop_last=False, shuffle=True)
            for P, y in proto_loader:
                Q = F(P)

                latents = pipe.prepare_latents(
                            Q.shape[0],
                            self.pipe_num_channels_latents,
                            self.pipe_height,
                            self.pipe_width,
                            self.pipe_prompt_embeds.dtype,
                            self.pipe_device,
                            self.pipe_generator,
                            Q,
                        )
                latent_vec = latents.view(latents.shape[0], -1)

                centroid_latents = Centroids(y)
                loss = self.MSEloss(latent_vec, centroid_latents)

                opt_F.zero_grad()
                opt_Centroids.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(F.parameters(), 100)
                torch.nn.utils.clip_grad_norm_(Centroids.parameters(), 100)
                opt_F.step()
                opt_Centroids.step()

        self.set_Centroids(uploaded_protos, F, Centroids)
        latent_centroids_vec = Centroids(self.classes_ids_tensor).detach().data
        latent_centroids = [torch.squeeze(latent.view(self.pipe_latent_shape)) for latent in latent_centroids_vec]

        save_item(F, self.role, 'F', self.save_folder_name)
        save_item(Centroids, self.role, 'Centroids', self.save_folder_name)
        save_item(latent_centroids, self.role, 'latent_centroids', self.save_folder_name)

    @torch.no_grad()
    def generate_images(self, R):
        print('\n----Server generating ...----\n')

        pipe = load_item(self.role, 'pipe', self.save_folder_name)
        latent_centroids = load_item(self.role, 'latent_centroids', self.save_folder_name)
        clientprocess = load_item(self.role, 'clientprocess', self.save_folder_name)

        pipe.scheduler.set_timesteps(self.pipe_num_inference_steps, device=self.pipe_device)
        timesteps = pipe.scheduler.timesteps

        for c in range(self.num_classes):

            latents = latent_centroids[c]
            latents = latents * pipe.scheduler.init_noise_sigma

            prompt_embeds = self.pipe_prompt_embeds.repeat_interleave(latents.shape[0], dim=0)
            negative_prompt_embeds = self.pipe_negative_prompt_embeds.repeat_interleave(latents.shape[0], dim=0)
            if self.pipe_do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            extra_step_kwargs = pipe.prepare_extra_step_kwargs(self.pipe_generator, self.pipe_eta)

            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.pipe_do_classifier_free_guidance else latents
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=self.pipe_cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.pipe_do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.pipe_guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.pipe_do_classifier_free_guidance and self.pipe_guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.pipe_guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            # image, has_nsfw_concept = pipe.run_safety_checker(image, self.pipe_device, prompt_embeds.dtype)
            do_denormalize = [True] * image.shape[0]
            image = pipe.image_processor.postprocess(image, output_type=self.pipe_output_type, do_denormalize=do_denormalize)
            # print(image[0])
            img_tensor = clientprocess(image[0])
            img_list = []
            for _ in range(self.gen_batch_size):
                img_list.append(img_tensor)
            img_list = torch.stack(img_list)

            save_item(img_list, 'server', f'image_{c}', self.save_folder_name, R=R)

    def train_global_model(self, round_idx):
        """Train global model using aggregated prototypes"""
        if not self.use_global_model:
            return

        print(f"\n==== Training Global Model (Round {round_idx}) ====")

        # Load components
        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)

        if self.use_etf:
            ETF = load_item(self.role, 'ETF', self.save_folder_name)
            ETF = F.normalize(ETF.T)

        # Create training data from prototypes
        proto_loader = DataLoader(uploaded_protos, self.server_batch_size,
                                 drop_last=False, shuffle=True)

        # Setup optimizer
        optimizer = torch.optim.SGD(global_model.parameters(),
                                   lr=self.server_learning_rate)
        global_model.train()

        # Training loop
        for epoch in range(min(10, self.server_epochs // 10)):  # Fewer epochs for global model
            epoch_loss = 0
            for P, y in proto_loader:
                P = P.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()

                # Forward pass through global model
                outputs = global_model(P)

                if self.use_etf:
                    # Compute loss with ETF classifier
                    logits = outputs @ ETF
                    loss = F.cross_entropy(logits, y)
                else:
                    # Standard cross-entropy loss
                    loss = F.cross_entropy(outputs, y)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss = {epoch_loss/len(proto_loader):.4f}")

        # Save updated global model
        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        print("Global model training completed")

    def generate_final_samples(self):
        """Generate final sample images for visualization"""
        print("\n----Generating final sample images----")
        # Implementation for final sample generation
        pass


class Feature_Transformer(nn.Module):
    def __init__(self, num_classes, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(num_classes, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        return out


def generate_random_orthogonal_matrix(dim_in, dim_out):
    # 生成正交矩阵
    H = np.random.randn(dim_in, dim_in).astype(np.float32)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    P = torch.tensor(u[:, :dim_out]).to(torch.float)
    return P


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def avg_func(protos):
    """Average prototypes per class"""
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = torch.stack(proto_list).mean(0)
            protos[label] = proto
        else:
            protos[label] = proto_list[0]
    return protos