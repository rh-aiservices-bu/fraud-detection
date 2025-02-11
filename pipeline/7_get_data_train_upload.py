import os

from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath

from kfp import kubernetes


@dsl.component(base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.11-2024b-20241108")
def get_data(train_data_output_path: OutputPath(), validate_data_output_path: OutputPath()):
    import urllib.request
    print("starting download...")
    print("downloading training data")
    url = "https://raw.githubusercontent.com/cfchase/fraud-detection/main/data/train.csv"
    urllib.request.urlretrieve(url, train_data_output_path)
    print("train data downloaded")
    print("downloading validation data")
    url = "https://raw.githubusercontent.com/cfchase/fraud-detection/main/data/validate.csv"
    urllib.request.urlretrieve(url, validate_data_output_path)
    print("validation data downloaded")


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.11-2024b-20241108",
    packages_to_install=["onnx", "onnxruntime", "onnxscript"],
)
def train_model(train_data_input_path: InputPath(), validate_data_input_path: InputPath(), model_output_path: OutputPath()):
    import torch
    import pandas as pd 
    from torch.utils.data import Dataset, DataLoader
    from torch import nn
    from sklearn.metrics import precision_score, recall_score
    import os

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

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

    train_df = pd.read_csv(train_data_input_path)
    labels_df = train_df.iloc[:, label_indexes]
    train_df = train_df.iloc[:, feature_indexes]
    train_df_tensor = torch.tensor(train_df.values, dtype=torch.float).to(device)
    labels_df_tensor = torch.tensor(labels_df.values, dtype=torch.float).to(device)

    # like scikit learn standard scaler
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


    train_df_tensor = torch.tensor(train_df.values, dtype=torch.float).to(device)
    scaler = TorchStandardScaler()
    scaler.fit(train_df_tensor)
    scaler.mean, scaler.std

    class CSVDataset(Dataset):
        def __init__(self, csv_file, pyarrow_fs=None, transform=None, target_transform=None):
            self.feature_indexes = feature_indexes
            self.label_indexes = label_indexes
            
            if pyarrow_fs:
                with pyarrow_fs.open_input_file(csv_file) as file:
                    training_table = pv.read_csv(file)
                self.data = training_table.to_pandas()
            else:
                self.data = pd.read_csv(csv_file)


            self.features = self.data.iloc[:, self.feature_indexes].values
            self.labels = self.data.iloc[:, self.label_indexes].values
            self.features = torch.tensor(self.features, dtype=torch.float).to(device)
            self.labels = torch.tensor(self.labels, dtype=torch.float).to(device)

            self.transform = transform
            self.target_transform = target_transform

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


    training_data = CSVDataset('data/train.csv')
    validation_data = CSVDataset(train_data_input_path)

    batch_size = 64

    training_dataloader = DataLoader(training_data, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    class NeuralNetwork(nn.Module):
        def __init__(self, scaler):
            super().__init__()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(5, 32),
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


    model = NeuralNetwork(scaler).to(device)

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

    loss_fn = nn.BCELoss().to(device)

    learning_rate = 1e-3
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    num_epochs = 2
    for t in range(num_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_loop(training_dataloader, model, loss_fn, optimizer)
        metrics = eval_loop(validation_dataloader, model, loss_fn)
        print(f"Eval Metrics: \n Accuracy: {(metrics['accuracy']):>0.1f}%, Avg loss: {metrics['loss']:>8f}, "
            f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f} \n")

    dummy_input = torch.randn(1, 5, device=device)
    onnx_model = torch.onnx.export(
        model,
        dummy_input,
        model_output_path,
        input_names=["inputs"],
        output_names=["outputs"],
        dynamic_axes={
            "inputs": {0: "batch_size"},
        },
        verbose=True)

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-pytorch-ubi9-python-3.11-2024b-20241108",
    packages_to_install=["boto3", "botocore"]
)
def upload_model(input_model_path: InputPath()):
    import os
    import boto3
    import botocore

    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    endpoint_url = os.environ.get('AWS_S3_ENDPOINT')
    region_name = os.environ.get('AWS_DEFAULT_REGION')
    bucket_name = os.environ.get('AWS_S3_BUCKET')

    s3_key = os.environ.get("S3_KEY")

    session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                    aws_secret_access_key=aws_secret_access_key)

    s3_resource = session.resource(
        's3',
        config=botocore.client.Config(signature_version='s3v4'),
        endpoint_url=endpoint_url,
        region_name=region_name)

    bucket = s3_resource.Bucket(bucket_name)

    print(f"Uploading {s3_key}")
    bucket.upload_file(input_model_path, s3_key)


@dsl.pipeline(name=os.path.basename(__file__).replace('.py', ''))
def pipeline():
    get_data_task = get_data()
    train_data_csv_file = get_data_task.outputs["train_data_output_path"]
    validate_data_csv_file = get_data_task.outputs["validate_data_output_path"]

    train_model_task = train_model(train_data_input_path=train_data_csv_file,
                                   validate_data_input_path=validate_data_csv_file)
    onnx_file = train_model_task.outputs["model_output_path"]

    upload_model_task = upload_model(input_model_path=onnx_file)

    upload_model_task.set_env_variable(name="S3_KEY", value="models/fraud/1/model.onnx")

    kubernetes.use_secret_as_env(
        task=upload_model_task,
        secret_name='aws-connection-my-storage',
        secret_key_to_env={
            'AWS_ACCESS_KEY_ID': 'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY': 'AWS_SECRET_ACCESS_KEY',
            'AWS_DEFAULT_REGION': 'AWS_DEFAULT_REGION',
            'AWS_S3_BUCKET': 'AWS_S3_BUCKET',
            'AWS_S3_ENDPOINT': 'AWS_S3_ENDPOINT',
        })


if __name__ == '__main__':
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=__file__.replace('.py', '.yaml')
    )
