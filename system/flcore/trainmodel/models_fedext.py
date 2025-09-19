import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.models import CNN, FedAvgCNN, FedAvgMLP, DNN, LeNet
import torchvision


class FedEXTModel(nn.Module):
    """FedEXT model with configurable encoder/classifier split"""

    def __init__(self, args, cid):
        super().__init__()

        # Get encoder split ratio (default 0.7 means 70% encoder, 30% classifier)
        self.encoder_ratio = getattr(args, 'encoder_ratio', 0.7)

        # Store dimensions
        self.feature_dim = args.feature_dim
        self.num_classes = args.num_classes

        # Get model name
        model_name = args.models[cid % len(args.models)] if hasattr(args, 'models') else args.model

        # Build model with encoder/classifier split based on ratio
        self._build_split_model(model_name, args)

    def _build_split_model(self, model_name, args):
        """Build encoder and classifier by splitting model layers based on encoder_ratio"""

        # First, create the full model to analyze its structure
        if 'FedAvgCNN' in model_name:
            self._build_fedavgcnn_split(args)
        elif 'FedAvgMLP' in model_name:
            self._build_fedavgmlp_split(args)
        elif 'resnet' in model_name.lower():
            self._build_resnet_split(model_name, args)
        elif 'googlenet' in model_name.lower():
            self._build_googlenet_split(args)
        elif 'mobilenet' in model_name.lower():
            self._build_mobilenet_split(args)
        elif 'vit' in model_name.lower():
            self._build_vit_split(model_name, args)
        elif 'CNN' in model_name:
            self._build_cnn_split(args)
        elif 'LeNet' in model_name:
            self._build_lenet_split(args)
        else:
            # Default: Simple MLP
            self._build_default_split(args)

    def _build_fedavgcnn_split(self, args):
        """Split FedAvgCNN based on encoder_ratio"""
        in_features = 1 if 'MNIST' in args.dataset else 3
        dim = 1024 if 'MNIST' in args.dataset else 1600

        # Total layers: conv1, conv2, flatten, fc1, fc
        # With encoder_ratio=0.7, encoder gets: conv1, conv2, flatten (60% of compute)
        # Plus a projection layer to feature_dim

        self.encoder = nn.Sequential(
            # Conv block 1
            nn.Conv2d(in_features, 32, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=5, padding=0, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            # Flatten and project to feature dimension
            nn.Flatten(),
            nn.Linear(dim, self.feature_dim),
            nn.ReLU(inplace=True)
        )

        # Classifier gets the remaining layers
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_classes)
        )

    def _build_fedavgmlp_split(self, args):
        """Split FedAvgMLP based on encoder_ratio"""
        in_features = 784 if 'MNIST' in args.dataset else 3072

        # Calculate hidden dimensions based on encoder_ratio
        # Original model: input -> 200 -> num_classes
        # Split: input -> hidden1 -> feature_dim -> num_classes
        hidden1 = int(200 * self.encoder_ratio) if self.encoder_ratio > 0.5 else 100

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden1),
            nn.ReLU(inplace=True),
            nn.Linear(hidden1, self.feature_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _build_resnet_split(self, model_name, args):
        """Split ResNet models based on encoder_ratio"""
        # Create the base ResNet model
        if 'resnet18' in model_name.lower():
            base_model = torchvision.models.resnet18(pretrained=False, num_classes=self.num_classes)
            total_layers = 8  # conv1 + 4 blocks (each with 2 layers)
        elif 'resnet34' in model_name.lower():
            base_model = torchvision.models.resnet34(pretrained=False, num_classes=self.num_classes)
            total_layers = 16  # conv1 + 4 blocks
        elif 'resnet50' in model_name.lower():
            base_model = torchvision.models.resnet50(pretrained=False, num_classes=self.num_classes)
            total_layers = 16  # conv1 + 4 blocks (bottleneck)
        elif 'resnet101' in model_name.lower():
            base_model = torchvision.models.resnet101(pretrained=False, num_classes=self.num_classes)
            total_layers = 33  # conv1 + 4 blocks
        elif 'resnet152' in model_name.lower():
            base_model = torchvision.models.resnet152(pretrained=False, num_classes=self.num_classes)
            total_layers = 50  # conv1 + 4 blocks
        else:
            # Default ResNet18
            base_model = torchvision.models.resnet18(pretrained=False, num_classes=self.num_classes)
            total_layers = 8

        # Calculate split point based on encoder_ratio
        encoder_layers = int(total_layers * self.encoder_ratio)

        # For ResNet, we split at block level
        # encoder_ratio=0.7 means encoder gets conv1 + layer1 + layer2 + part of layer3
        if self.encoder_ratio <= 0.25:
            # Only conv1 and layer1
            encoder_modules = [
                base_model.conv1, base_model.bn1, base_model.relu,
                base_model.maxpool, base_model.layer1
            ]
        elif self.encoder_ratio <= 0.5:
            # conv1, layer1, layer2
            encoder_modules = [
                base_model.conv1, base_model.bn1, base_model.relu,
                base_model.maxpool, base_model.layer1, base_model.layer2
            ]
        elif self.encoder_ratio <= 0.75:
            # conv1, layer1, layer2, layer3
            encoder_modules = [
                base_model.conv1, base_model.bn1, base_model.relu,
                base_model.maxpool, base_model.layer1, base_model.layer2,
                base_model.layer3
            ]
        else:
            # Almost everything except final fc
            encoder_modules = [
                base_model.conv1, base_model.bn1, base_model.relu,
                base_model.maxpool, base_model.layer1, base_model.layer2,
                base_model.layer3, base_model.layer4
            ]

        # Build encoder with adaptive pooling and projection
        self.encoder = nn.Sequential(
            *encoder_modules,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(base_model.fc.in_features if self.encoder_ratio > 0.75
                     else (256 if self.encoder_ratio > 0.5
                           else (128 if self.encoder_ratio > 0.25 else 64)),
                     self.feature_dim),
            nn.ReLU(inplace=True)
        )

        # Classifier
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _build_googlenet_split(self, args):
        """Split GoogLeNet based on encoder_ratio"""
        base_model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                  num_classes=self.num_classes)

        # GoogLeNet has inception modules
        # Split at inception module level based on encoder_ratio
        if self.encoder_ratio <= 0.5:
            # Up to inception3b
            encoder_modules = [
                base_model.conv1, base_model.maxpool1, base_model.conv2,
                base_model.conv3, base_model.maxpool2, base_model.inception3a,
                base_model.inception3b, base_model.maxpool3
            ]
            out_channels = 480
        elif self.encoder_ratio <= 0.75:
            # Up to inception4d
            encoder_modules = [
                base_model.conv1, base_model.maxpool1, base_model.conv2,
                base_model.conv3, base_model.maxpool2, base_model.inception3a,
                base_model.inception3b, base_model.maxpool3, base_model.inception4a,
                base_model.inception4b, base_model.inception4c, base_model.inception4d
            ]
            out_channels = 528
        else:
            # Almost all inception modules
            encoder_modules = [
                base_model.conv1, base_model.maxpool1, base_model.conv2,
                base_model.conv3, base_model.maxpool2, base_model.inception3a,
                base_model.inception3b, base_model.maxpool3, base_model.inception4a,
                base_model.inception4b, base_model.inception4c, base_model.inception4d,
                base_model.inception4e, base_model.maxpool4, base_model.inception5a,
                base_model.inception5b
            ]
            out_channels = 1024

        self.encoder = nn.Sequential(
            *encoder_modules,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, self.feature_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _build_mobilenet_split(self, args):
        """Split MobileNetV2 based on encoder_ratio"""
        from flcore.trainmodel.mobilenet_v2 import mobilenet_v2
        base_model = mobilenet_v2(pretrained=False, num_classes=self.num_classes)

        # MobileNetV2 has 17 inverted residual blocks
        # Split based on encoder_ratio
        total_blocks = len(base_model.features)
        encoder_blocks = int(total_blocks * self.encoder_ratio)

        encoder_modules = base_model.features[:encoder_blocks]

        # Determine output channels based on split point
        if encoder_blocks <= 7:
            out_channels = 32
        elif encoder_blocks <= 11:
            out_channels = 64
        elif encoder_blocks <= 14:
            out_channels = 96
        else:
            out_channels = 320

        self.encoder = nn.Sequential(
            encoder_modules,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(out_channels, self.feature_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _build_vit_split(self, model_name, args):
        """Split Vision Transformer based on encoder_ratio"""
        if 'vit_b_16' in model_name.lower():
            base_model = torchvision.models.vit_b_16(image_size=32, num_classes=self.num_classes)
            total_blocks = 12
        else:  # vit_b_32
            base_model = torchvision.models.vit_b_32(image_size=32, num_classes=self.num_classes)
            total_blocks = 12

        # Split transformer blocks based on encoder_ratio
        encoder_blocks = int(total_blocks * self.encoder_ratio)

        # ViT structure: patch_embed -> encoder blocks -> head
        # We'll take encoder_blocks number of transformer blocks
        encoder_list = [
            base_model.conv_proj,  # Patch embedding
            base_model.encoder.pos_embedding,
            base_model.encoder.dropout
        ]

        # Add the calculated number of encoder blocks
        for i in range(encoder_blocks):
            encoder_list.append(base_model.encoder.layers[i])

        # Create a custom encoder module
        class ViTEncoder(nn.Module):
            def __init__(self, modules, feature_dim):
                super().__init__()
                self.initial = nn.Sequential(*modules[:3])
                self.blocks = nn.Sequential(*modules[3:])
                self.norm = nn.LayerNorm(768)
                self.proj = nn.Linear(768, feature_dim)

            def forward(self, x):
                x = self.initial[0](x)  # conv_proj
                x = x.flatten(2).transpose(1, 2)  # Reshape
                x = self.initial[1](x)  # pos_embedding
                x = self.initial[2](x)  # dropout
                x = self.blocks(x)
                x = self.norm(x)
                x = x[:, 0]  # Take CLS token
                x = self.proj(x)
                return x

        self.encoder = ViTEncoder(encoder_list, self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _build_cnn_split(self, args):
        """Split generic CNN based on encoder_ratio"""
        in_features = 1 if 'MNIST' in args.dataset else 3

        # Similar to FedAvgCNN
        self._build_fedavgcnn_split(args)

    def _build_lenet_split(self, args):
        """Split LeNet based on encoder_ratio"""
        # LeNet layers: conv1 -> pool -> conv2 -> pool -> fc1 -> fc2 -> fc3
        # With encoder_ratio=0.7: encoder gets conv layers + first fc

        if self.encoder_ratio <= 0.5:
            # Only conv layers
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(50*4*4, self.feature_dim),
                nn.ReLU()
            )
        else:
            # Conv layers + some FC layers
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(50*4*4, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, self.feature_dim),
                nn.ReLU()
            )

        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def _build_default_split(self, args):
        """Default split for unknown models"""
        in_features = 784 if 'MNIST' in args.dataset else 3072

        # Simple MLP with split based on encoder_ratio
        hidden1 = int(256 * self.encoder_ratio) if self.encoder_ratio > 0.5 else 128

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, self.feature_dim),
            nn.ReLU()
        )

        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

    def forward(self, x, return_features=False):
        """
        Forward pass through the model
        Args:
            x: input data
            return_features: if True, returns features instead of predictions
        """
        features = self.encoder(x)

        if return_features:
            return features

        output = self.classifier(features)
        return output

    def get_encoder_params(self):
        """Get encoder parameters for aggregation"""
        return self.encoder.state_dict()

    def get_classifier_params(self):
        """Get classifier parameters for group aggregation"""
        return self.classifier.state_dict()

    def set_encoder_params(self, state_dict):
        """Set encoder parameters"""
        self.encoder.load_state_dict(state_dict)

    def set_classifier_params(self, state_dict):
        """Set classifier parameters"""
        self.classifier.load_state_dict(state_dict)

    def extract_features(self, dataloader, device):
        """
        Extract features and labels from dataloader
        Returns:
            features: torch.Tensor of extracted features
            labels: torch.Tensor of corresponding labels
        """
        self.eval()
        all_features = []
        all_labels = []

        with torch.no_grad():
            for x, y in dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)

                # Extract features using encoder
                features = self.encoder(x)

                all_features.append(features.cpu())
                all_labels.append(y.cpu())

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)