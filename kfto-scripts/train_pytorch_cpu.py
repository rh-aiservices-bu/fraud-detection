def train_func():
    import pandas as pd
    import numpy as np
    from numpy.typing import NDArray
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import os
    import logging
    from typing import Tuple, List

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.onnx
    import boto3

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    def setup_ddp() -> Tuple[int, int, int, torch.device]:
        """Initialize the distributed training environment."""
        try:
            rank = int(os.environ.get("RANK", 0))
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))

            # Initialize the distributed environment
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

            device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

            if not torch.cuda.is_available():
                device = torch.device("cpu")

            return rank, local_rank, world_size, device
        except Exception as e:
            logger.error(f"Failed to setup DDP: {e}")
            raise

    # Initialize DDP and get process-specific information
    rank, local_rank, world_size, device = setup_ddp()

    if rank == 0:
        logger.info(f"DDP setup: Rank {rank}/{world_size} on device {device}")

    # Data paths from environment variables with validation
    bucket_name = os.environ.get("AWS_S3_BUCKET")
    if not bucket_name:
        raise ValueError("AWS_S3_BUCKET environment variable is required")

    train_data = os.environ.get("TRAIN_DATA", "data/train.csv")
    test_data = os.environ.get("TEST_DATA", "data/test.csv")

    train_file_uri = f"s3://{bucket_name}/{train_data}"
    test_file_uri = f"s3://{bucket_name}/{test_data}"

    class FraudDataset(Dataset):
        def __init__(self, features: NDArray[np.float32], labels: NDArray[np.float32]):
            self.features = torch.tensor(features, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.features[idx], self.labels[idx]

    def load_and_preprocess_data(file_path: str) -> Tuple[NDArray[np.float32], NDArray[np.float32], StandardScaler]:
        try:
            df = pd.read_csv(
                file_path,
                storage_options={
                    # Added "verify": False flag to suport using insecure S3 buckets
                    "client_kwargs": {"endpoint_url": os.environ.get("AWS_S3_ENDPOINT"), "verify": False}
                }
            )

            feature_columns = [
                'distance_from_last_transaction',
                'ratio_to_median_purchase_price',
                'used_chip',
                'used_pin_number',
                'online_order'
            ]

            X = df[feature_columns].to_numpy().astype(np.float32)
            y = df['fraud'].to_numpy().astype(np.float32)

            scaler = StandardScaler()
            # Only scale numerical features (first two columns)
            X[:, :2] = scaler.fit_transform(X[:, :2])

            return X, y, scaler
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
            raise

    # Load and preprocess data
    if rank == 0:
        logger.info("Loading and preprocessing data...")

    X_train, y_train, scaler_train = load_and_preprocess_data(train_file_uri)
    X_test, y_test, scaler_test = load_and_preprocess_data(test_file_uri)

    if rank == 0:
        logger.info("Data loading and preprocessing complete.")

    # Create PyTorch Datasets
    train_dataset = FraudDataset(X_train, y_train)
    test_dataset = FraudDataset(X_test, y_test)

    # Calculate pos_weight for handling class imbalance
    num_pos = np.sum(y_train == 1).astype(np.float32)
    num_neg = np.sum(y_train == 0).astype(np.float32)

    if num_pos > 0 and num_neg > 0:
        pos_weight_value = torch.tensor(num_neg / num_pos, dtype=torch.float32).to(device)
        if rank == 0:
            logger.info(f"Calculated pos_weight: {pos_weight_value.item():.4f} (num_neg={num_neg}, num_pos={num_pos})")
    else:
        pos_weight_value = torch.tensor(1.0, dtype=torch.float32).to(device)
        if rank == 0:
            logger.warning("No positive or negative samples found. Using default pos_weight=1.0")

    # Create Distributed Samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders
    batch_size = 64
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=4 if torch.cuda.is_available() else 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        pin_memory=True if torch.cuda.is_available() else False,
        num_workers=4 if torch.cuda.is_available() else 0
    )

    class FraudDetector(nn.Module):
        def __init__(self, input_dim: int):
            super(FraudDetector, self).__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.fc4 = nn.Linear(32, 1)

            # Initialize weights
            for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.relu(x)
            x = self.fc4(x)
            return x

    # Initialize model
    input_dim = X_train.shape[1]
    model = FraudDetector(input_dim).to(device)

    # Wrap model with DDP
    if device.type == 'cuda':
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)

    if rank == 0:
        logger.info(f"Model Architecture:\n{model.module}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # Training Loop
    num_epochs = 3

    if rank == 0:
        logger.info("Starting Training")

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (features, labels) in enumerate(train_loader):
            try:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(features)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item() * features.size(0)
                preds = torch.round(torch.sigmoid(outputs))
                correct_predictions += (preds == labels).sum().item()
                total_samples += labels.size(0)

            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue

        # Aggregate metrics across processes
        metrics = torch.tensor([running_loss, correct_predictions, total_samples], dtype=torch.float64).to(device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        running_loss, correct_predictions, total_samples = metrics.tolist()

        if rank == 0:
            epoch_loss = running_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")
            scheduler.step(epoch_loss)

    if rank == 0:
        logger.info("Training Complete")

    # Evaluation
    if rank == 0:
        logger.info("Starting Evaluation")

    model.eval()
    all_labels: List[float] = []
    all_predictions: List[float] = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            probabilities = torch.sigmoid(outputs)
            predictions = torch.round(probabilities)

            # Convert to float type explicitly
            all_labels.extend(labels.cpu().numpy().flatten().astype(float).tolist())
            all_predictions.extend(predictions.cpu().numpy().flatten().astype(float).tolist())

    # Gather predictions from all processes
    gathered_labels: List[List[float]] = [[] for _ in range(world_size)]
    gathered_predictions: List[List[float]] = [[] for _ in range(world_size)]

    dist.all_gather_object(gathered_labels, all_labels)
    dist.all_gather_object(gathered_predictions, all_predictions)

    if rank == 0:
        final_labels = np.array(sum(gathered_labels, []), dtype=np.float32)
        final_predictions = np.array(sum(gathered_predictions, []), dtype=np.float32)

        metrics = {
            'accuracy': accuracy_score(final_labels, final_predictions),
            'precision': precision_score(final_labels, final_predictions),
            'recall': recall_score(final_labels, final_predictions),
            'f1': f1_score(final_labels, final_predictions)
        }

        logger.info("\nEvaluation Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name.capitalize()}: {value:.4f}")

        conf_matrix = confusion_matrix(final_labels, final_predictions)
        logger.info(f"\nConfusion Matrix:\n{conf_matrix}")
        logger.info("  [[TN, FP]\n   [FN, TP]]")

        # Save PyTorch model
        model_save_path = 'fraud_detector_model.pth'
        torch.save(model.module.state_dict(), model_save_path)
        logger.info(f"\nModel saved to {model_save_path}")

        # Export to ONNX
        try:
            onnx_model_path = 'fraud_detector_model.onnx'
            export_model = FraudDetector(input_dim)
            export_model.load_state_dict(torch.load(model_save_path, map_location='cpu'))
            export_model.eval()

            dummy_input = torch.randn(1, input_dim)

            # Export to ONNX
            torch.onnx.export(
                export_model,
                (dummy_input,),  # Wrap input in a tuple
                onnx_model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"Model exported to ONNX format: {onnx_model_path}")

            # Upload to S3
            try:
                # Added verify=False flag to suport using insecure S3 buckets
                s3_client = boto3.client('s3', endpoint_url=os.environ.get("AWS_S3_ENDPOINT"), verify=False)
                s3_key = f"models/{os.path.basename(onnx_model_path)}"
                s3_client.upload_file(onnx_model_path, bucket_name, s3_key)
                logger.info(f"Model uploaded to s3://{bucket_name}/{s3_key}")
            except Exception as s3_e:
                logger.error(f"Failed to upload model to S3: {s3_e}")

        except Exception as e:
            logger.error(f"Failed to export/upload model: {e}")

    # Cleanup
    dist.destroy_process_group()
    if rank == 0:
        logger.info("DDP process group destroyed.")
