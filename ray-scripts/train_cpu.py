import os
import torch
import torch.nn as nn
import ray
import pandas as pd
import tempfile
import boto3
import botocore

from sklearn.metrics import precision_score, recall_score

import pyarrow
import pyarrow.fs
import pyarrow.csv

from torch.utils.data import Dataset, DataLoader
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig

feature_indexes = [
    1,  # distance_from_last_transaction
    2,  # ratio_to_median_purchase_price
    4,  # used_chip
    5,  # used_pin_number
    6,  # online_order
]

label_indexes = [
    7  # fraud
]

first_layer_num = len(feature_indexes)

device = "cpu"
use_gpu = False
num_epochs = 1
batch_size = 64
bucket_name = os.environ.get("AWS_S3_BUCKET")
state_dict_filename = "state_dict.pth"
full_model_filename = "model.pth"


class TorchStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor):
        self.mean = tensor.mean(dim=0, keepdim=False)
        self.std = tensor.std(dim=0, keepdim=False)

    def transform(self, tensor):
        return (tensor - self.mean) / self.std

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)


class CSVDataset(Dataset):
    def __init__(self, csv_file, pyarrow_fs=None, transform=None, target_transform=None):
        self.feature_indexes = feature_indexes
        self.label_indexes = label_indexes

        if pyarrow_fs:
            with pyarrow_fs.open_input_file(csv_file) as file:
                training_table = pyarrow.csv.read_csv(file)
            self.data = training_table.to_pandas()
        else:
            self.data = pd.read_csv(csv_file)

        self.features = self.data.iloc[:, self.feature_indexes].values
        self.labels = self.data.iloc[:, self.label_indexes].values
        self.features = torch.tensor(self.features, dtype=torch.float).to(device)
        self.labels = torch.tensor(self.labels, dtype=torch.float).to(device)

        self.transform = transform
        self.target_transform = target_transform

        # small dataset will fit and
        if self.transform:
            self.features = self.transform(self.features)
        if self.target_transform:
            self.labels = self.target_transform(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.features[idx]
        label = self.labels[idx]
        return features, label


class NeuralNetwork(nn.Module):
    def __init__(self, scaler):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(first_layer_num, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.scaler = scaler

    def forward(self, x):
        with torch.no_grad():
            x_pre = self.scaler.transform(x)
        probs = self.linear_relu_stack(x_pre)
        return probs


def get_scaler(pyarrow_fs):
    train_csv_path = bucket_name + "/" + os.environ.get("TRAIN_DATA")
    with pyarrow_fs.open_input_file(train_csv_path) as file:
        training_table = pyarrow.csv.read_csv(file)

    train_df = training_table.to_pandas()
    train_df = train_df.iloc[:, feature_indexes]
    train_df_tensor = torch.tensor(train_df.values, dtype=torch.float).to(device)
    scaler = TorchStandardScaler()
    scaler.fit(train_df_tensor)

    return scaler


def get_datasets(pyarrow_fs):
    train_csv_path = bucket_name + "/" + os.environ.get("TRAIN_DATA")
    validate_csv_path = bucket_name + "/" + os.environ.get("VALIDATE_DATA")

    training_data = CSVDataset(train_csv_path, pyarrow_fs)
    validation_data = CSVDataset(validate_csv_path, pyarrow_fs)

    return training_data, validation_data


def get_fs():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
    region_name = os.environ.get("AWS_DEFAULT_REGION")

    return pyarrow.fs.S3FileSystem(
        access_key=aws_access_key_id,
        secret_key=aws_secret_access_key,
        region=region_name,
        endpoint_override=endpoint_url)

def get_s3_resource():
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    return s3_resource


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % round(size / batch_size / 10) == 0:
            loss = loss.item()
            current = batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def eval_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    eval_loss, correct = 0, 0

    all_preds = torch.tensor([])
    all_labels = torch.tensor([])

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            correct += torch.eq(torch.round(pred), y).sum().item()

            pred_labels = torch.round(pred)
            all_preds = torch.cat((all_preds, pred_labels.cpu()))
            all_labels = torch.cat((all_labels, y.cpu()))

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)

    eval_loss /= num_batches
    accuracy = correct / size * 100

    return {
        "accuracy": accuracy,
        "loss": eval_loss,
        "precision": precision,
        "recall": recall
    }


def train_func_distributed():
    pyarrow_fs = get_fs()

    training_data, validation_data = get_datasets(pyarrow_fs)
    training_dataloader = DataLoader(training_data, batch_size=batch_size)
    training_dataloader = ray.train.torch.prepare_data_loader(training_dataloader)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)
    validation_dataloader = ray.train.torch.prepare_data_loader(validation_dataloader)

    model = NeuralNetwork(scaler).to(device)
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.BCELoss().to(device)
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        if ray.train.get_context().get_world_size() > 1:
            training_dataloader.sampler.set_epoch(epoch)

        train_loop(training_dataloader, model, loss_fn, optimizer)
        metrics = eval_loop(validation_dataloader, model, loss_fn)
        metrics["epoch"] = epoch

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            torch.save(
                model.module.state_dict(),
                os.path.join(temp_checkpoint_dir, state_dict_filename)
            )
            ray.train.report(
                metrics,
                checkpoint=ray.train.Checkpoint.from_directory(temp_checkpoint_dir),
            )
        if ray.train.get_context().get_world_rank() == 0:
            print(metrics)


def save_full_model(checkpoint_path):
    s3_resource = get_s3_resource()
    bucket = s3_resource.Bucket(bucket_name)

    cp_s3_key = checkpoint_path.removeprefix(f"{bucket_name}/") + "/" + state_dict_filename
    state_dict_local = f"/tmp/{state_dict_filename}"
    print(f"Downloading model state_dict from {cp_s3_key} to {state_dict_local}")
    bucket.download_file(cp_s3_key, state_dict_local)

    upload_path = os.environ.get("MODEL_OUTPUT")
    upload_state_dict_s3_key = os.path.join(upload_path, state_dict_filename)
    bucket.upload_file(state_dict_local, upload_state_dict_s3_key)


    full_model = NeuralNetwork(scaler)
    full_model.load_state_dict(torch.load(state_dict_local))
    full_model_local = f"/tmp/{full_model_filename}"
    torch.save(full_model, full_model_local)

    upload_model_s3_key = os.path.join(upload_path, full_model_filename)
    print(f"Uploading model from {full_model_local} to {upload_model_s3_key}")
    bucket.upload_file(full_model_local, upload_model_s3_key)


pyarrow_fs = get_fs()
scaler = get_scaler(pyarrow_fs)

trainer = TorchTrainer(
    train_func_distributed,
    run_config=RunConfig(
        storage_filesystem=pyarrow_fs,
        storage_path=f"{bucket_name}/ray/",
        name="fraud-training",
    ),
    scaling_config=ScalingConfig(
        num_workers=3,   # num_workers = number of worker nodes with the ray head node included
        use_gpu=use_gpu,
    ),
)

results = trainer.fit()
print(type(results))
print(results)

# download_model(pyarrow_fs, results.checkpoint.path, f"/tmp/{model_file_name}")
# upload_model(f"/tmp/{model_file_name}")
save_full_model(results.checkpoint.path)
