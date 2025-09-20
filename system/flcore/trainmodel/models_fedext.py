import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.trainmodel.models import BaseHeadSplit
import copy


class FedEXTModel(BaseHeadSplit):
    """
    FedEXT model - simply inherits from BaseHeadSplit
    The split is already done: base (feature extractor) + head (classifier)

    encoder_ratio controls aggregation strategy:
    - ratio = 1.0: 100% global aggregation (FedAvg)
    - ratio = 0.5: 50% global (base), 50% group (head)
    - ratio = 0.0: 100% group aggregation

    In our implementation:
    - base is always aggregated globally with weight encoder_ratio
    - head is always aggregated in groups with weight (1 - encoder_ratio)
    """

    def __init__(self, args, cid):
        super().__init__(args, cid)

        # Store encoder ratio
        self.encoder_ratio = getattr(args, 'encoder_ratio', 0.7)
        self.cid = cid

        # The model is already split by BaseHeadSplit:
        # - self.base: feature extractor (outputs feature_dim)
        # - self.head: classifier (feature_dim -> num_classes)

    def get_base_params(self):
        """Get base parameters for global aggregation"""
        return self.base.state_dict()

    def get_head_params(self):
        """Get head parameters for group aggregation"""
        return self.head.state_dict()

    def set_base_params(self, state_dict):
        """Set base parameters"""
        self.base.load_state_dict(state_dict)

    def set_head_params(self, state_dict):
        """Set head parameters"""
        self.head.load_state_dict(state_dict)

    def extract_features(self, x):
        """Extract features using base"""
        return self.base(x)