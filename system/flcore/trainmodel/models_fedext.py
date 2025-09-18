import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.models import BaseHeadSplit


class FedEXTModel(nn.Module):
    """FedEXT model that explicitly separates encoder and classifier"""

    def __init__(self, args, cid):
        super().__init__()

        # Initialize base model (using existing BaseHeadSplit)
        base_model = BaseHeadSplit(args, cid)

        # Encoder is the feature extractor part
        self.encoder = base_model.base

        # Classifier is the head part (MLP)
        self.classifier = base_model.head

        # Store feature dimension for later use
        self.feature_dim = args.feature_dim
        self.num_classes = args.num_classes

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