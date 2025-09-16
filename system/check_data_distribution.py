#!/usr/bin/env python3
"""
Script to check data distribution across all clients.
"""
import sys
import os
import numpy as np
from collections import Counter

def check_client_data_distribution(dataset='Cifar10', num_clients=3):
    """Check the class distribution for all clients."""

    print(f"\nChecking data distribution for {dataset} with {num_clients} clients:")
    print("="*60)

    for client_id in range(num_clients):
        # Load training data
        train_file = f'../dataset/{dataset}/train/{client_id}.npz'

        if not os.path.exists(train_file):
            print(f"Client {client_id}: File not found - {train_file}")
            continue

        with open(train_file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        labels = data['y']
        class_counts = Counter(labels)
        total_samples = len(labels)

        print(f"\nClient {client_id}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Class distribution:")

        # Check for all classes (0-9 for CIFAR-10)
        for c in range(10):
            count = class_counts.get(c, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            status = "❌ MISSING" if count == 0 else "⚠️  LOW" if count < 10 else "✅"
            print(f"    Class {c}: {count:4d} samples ({percentage:5.1f}%) {status}")

        missing_classes = [c for c in range(10) if c not in class_counts]
        if missing_classes:
            print(f"  ⚠️  Missing classes: {missing_classes}")

        # Check test data
        test_file = f'../dataset/{dataset}/test/{client_id}.npz'
        if os.path.exists(test_file):
            with open(test_file, 'rb') as f:
                test_data = np.load(f, allow_pickle=True)['data'].tolist()
            test_labels = test_data['y']
            test_total = len(test_labels)
            print(f"  Test samples: {test_total}")

    print("\n" + "="*60)
    print("Summary:")
    print("❌ MISSING: Class has no samples")
    print("⚠️  LOW: Class has fewer than 10 samples")
    print("✅: Class has sufficient samples")

if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'Cifar10'
    num_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    check_client_data_distribution(dataset, num_clients)