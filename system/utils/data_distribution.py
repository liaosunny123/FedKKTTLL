"""
Data distribution configuration system for heterogeneous federated learning.
"""
import json
import os
import torch
import numpy as np
from collections import defaultdict
import random


class DataDistributionManager:
    """Manages data distribution for clients based on JSON configuration."""

    def __init__(self, config_path=None):
        """
        Initialize the distribution manager.

        Args:
            config_path: Path to JSON configuration file
        """
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded data distribution config from {config_path}")
        else:
            print("No distribution config found, using original data")

    def filter_client_data(self, client_id, train_data, num_classes):
        """
        Filter training data for a client based on configuration.

        Args:
            client_id: ID of the client
            train_data: Original training data [(x, y), ...]
            num_classes: Total number of classes

        Returns:
            Filtered training data based on configuration
        """
        client_key = str(client_id)

        # If no config for this client, return original data
        if client_key not in self.config:
            return train_data

        client_config = self.config[client_key]

        # Parse configuration
        if "classes" in client_config:
            # Filter by allowed classes
            allowed_classes = set(client_config["classes"])
            filtered_data = [(x, y) for x, y in train_data if int(y.item()) in allowed_classes]

        elif "class_distribution" in client_config:
            # Use specific distribution
            filtered_data = self._apply_distribution(train_data, client_config["class_distribution"])

        elif "class_percentages" in client_config:
            # Use percentage-based distribution
            filtered_data = self._apply_percentages(train_data, client_config["class_percentages"])

        elif "iid" in client_config and client_config["iid"]:
            # IID distribution - sample uniformly from all classes
            filtered_data = self._make_iid(train_data, num_classes, client_config.get("samples", len(train_data)))

        else:
            # Default: return original data
            filtered_data = train_data

        # Apply sample limit if specified
        if "max_samples" in client_config:
            max_samples = client_config["max_samples"]
            if len(filtered_data) > max_samples:
                filtered_data = random.sample(filtered_data, max_samples)

        # Print distribution info
        self._print_distribution(client_id, filtered_data, num_classes)

        return filtered_data

    def _apply_distribution(self, train_data, distribution):
        """Apply specific sample counts per class."""
        # Group data by class
        class_data = defaultdict(list)
        for x, y in train_data:
            class_data[int(y.item())].append((x, y))

        filtered_data = []
        for class_id, count in distribution.items():
            class_id = int(class_id)
            if class_id in class_data:
                available = class_data[class_id]
                if count == -1:  # Use all samples
                    filtered_data.extend(available)
                elif count > len(available):
                    # Use all available and repeat some
                    filtered_data.extend(available)
                    filtered_data.extend(random.choices(available, k=count-len(available)))
                else:
                    filtered_data.extend(random.sample(available, min(count, len(available))))

        random.shuffle(filtered_data)
        return filtered_data

    def _apply_percentages(self, train_data, percentages):
        """Apply percentage-based sampling per class."""
        # Group data by class
        class_data = defaultdict(list)
        for x, y in train_data:
            class_data[int(y.item())].append((x, y))

        filtered_data = []
        for class_id, percentage in percentages.items():
            class_id = int(class_id)
            if class_id in class_data:
                available = class_data[class_id]
                num_samples = int(len(available) * percentage)
                if num_samples > 0:
                    filtered_data.extend(random.sample(available, min(num_samples, len(available))))

        random.shuffle(filtered_data)
        return filtered_data

    def _make_iid(self, train_data, num_classes, num_samples):
        """Create IID distribution."""
        # Group data by class
        class_data = defaultdict(list)
        for x, y in train_data:
            class_data[int(y.item())].append((x, y))

        # Sample uniformly from each class
        samples_per_class = num_samples // num_classes
        filtered_data = []

        for class_id in range(num_classes):
            if class_id in class_data:
                available = class_data[class_id]
                if len(available) >= samples_per_class:
                    filtered_data.extend(random.sample(available, samples_per_class))
                else:
                    filtered_data.extend(available)

        random.shuffle(filtered_data)
        return filtered_data

    def _print_distribution(self, client_id, data, num_classes):
        """Print the distribution of filtered data."""
        if not data:
            print(f"Client {client_id}: No data!")
            return

        # Count samples per class
        class_counts = defaultdict(int)
        for _, y in data:
            class_counts[int(y.item())] += 1

        print(f"\nClient {client_id} data distribution:")
        print(f"  Total samples: {len(data)}")
        print(f"  Classes present: {sorted(class_counts.keys())}")

        for c in range(num_classes):
            count = class_counts[c]
            if count > 0:
                print(f"    Class {c}: {count:4d} samples ({count/len(data)*100:5.1f}%)")
            else:
                print(f"    Class {c}:    0 samples (MISSING)")

        missing_classes = set(range(num_classes)) - set(class_counts.keys())
        if missing_classes:
            print(f"  ⚠️  Missing classes: {sorted(missing_classes)}")