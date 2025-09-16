# Data Distribution Configuration for Heterogeneous Federated Learning

This directory contains JSON configuration files that define how data should be distributed across clients in federated learning experiments.

## Usage

To use a distribution configuration, add the `-dc` argument when running the training:

```bash
python main.py -data Cifar10 -nc 3 -dc configs/distribution_missing_classes.json
```

## Configuration Format

The JSON files map client IDs (as strings) to their data distribution specifications.

### Configuration Options

1. **`classes`**: List of class labels this client should have
   ```json
   "0": {
     "classes": [0, 1, 2, 3]
   }
   ```

2. **`class_distribution`**: Exact number of samples per class
   ```json
   "0": {
     "class_distribution": {
       "0": 100,
       "1": 200,
       "2": 50
     }
   }
   ```

3. **`class_percentages`**: Percentage of available samples per class (0.0-1.0)
   ```json
   "0": {
     "class_percentages": {
       "0": 0.1,
       "1": 0.5,
       "2": 0.8
     }
   }
   ```

4. **`iid`**: Create IID distribution with uniform sampling
   ```json
   "0": {
     "iid": true,
     "samples": 1000
   }
   ```

5. **`max_samples`**: Limit total samples for this client
   ```json
   "0": {
     "max_samples": 500
   }
   ```

## Available Configurations

### 1. `distribution_missing_classes.json`
- **Scenario**: Extreme label skew where clients have non-overlapping classes
- **Client 0**: Classes 3,4,5,7 (missing 0,1,2,6,8,9)
- **Client 1**: Classes 0,1,2,6,8,9 (complementary to Client 0)
- **Client 2**: IID distribution with all classes

### 2. `distribution_dirichlet.json`
- **Scenario**: Dirichlet distribution with α=0.5
- **Client 0**: Heavily skewed to classes 3,4,5
- **Client 1**: Heavily skewed to classes 0,1,2
- **Client 2**: Heavily skewed to classes 6,7,8,9

### 3. `distribution_label_skew.json`
- **Scenario**: Each client only has K=2 classes
- **Requires**: 5 clients for CIFAR-10
- **Usage**: `python main.py -nc 5 -dc configs/distribution_label_skew.json`

### 4. `distribution_quantity_skew.json`
- **Scenario**: Clients have different amounts of data
- **Client 0**: 100 samples (data-poor)
- **Client 1**: 500 samples (moderate)
- **Client 2**: 5000 samples (data-rich)

### 5. `distribution_original_problem.json`
- **Scenario**: Reproduces the original problem with missing classes
- **Client 0**: Missing classes 0,2,8 (as in the original issue)
- **Clients 1,2**: Use original data

### 6. `distribution_fix_missing.json`
- **Scenario**: Fixes missing classes with minimal synthetic data
- **Client 0**: Adds 50 samples for each missing class
- **Clients 1,2**: Use original data

## Creating Custom Configurations

Create a new JSON file with your desired distribution:

```json
{
  "0": {
    "classes": [0, 1, 2],
    "max_samples": 1000,
    "_comment": "Client 0 description"
  },
  "1": {
    "class_distribution": {
      "3": 500,
      "4": 500,
      "5": 500
    },
    "_comment": "Client 1 description"
  },
  "2": {
    "iid": true,
    "samples": 2000,
    "_comment": "Client 2 description"
  }
}
```

## Notes

- If a client is not specified in the config, it uses the original data
- The `_comment` field is ignored and can be used for documentation
- Configurations are applied after loading the original data
- The system will print the actual distribution for each client during initialization