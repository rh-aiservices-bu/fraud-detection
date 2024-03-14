import os

from kfp import compiler
from kfp import dsl
from kfp.dsl import InputPath, OutputPath

from kfp import kubernetes


@dsl.component(base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301")
def get_data(data_output_path: OutputPath()):
    import urllib.request
    print("starting download...")
    url = "https://raw.githubusercontent.com/rh-aiservices-bu/fraud-detection/main/data/card_transdata.csv"
    urllib.request.urlretrieve(url, data_output_path)
    print("done")

@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
    packages_to_install=["tf2onnx", "seaborn"],
)
def train_model(data_input_path: InputPath(), model_output_path: OutputPath()):
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, BatchNormalization, Activation
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.utils import class_weight
    import tf2onnx
    import onnx
    import pickle
    from pathlib import Path

    # Load the CSV data which we will use to train the model.
    # It contains the following fields:
    #   distancefromhome - The distance from home where the transaction happened.
    #   distancefromlast_transaction - The distance from last transaction happened.
    #   ratiotomedianpurchaseprice - Ratio of purchased price compared to median purchase price.
    #   repeat_retailer - If it's from a retailer that already has been purchased from before.
    #   used_chip - If the (credit card) chip was used.
    #   usedpinnumber - If the PIN number was used.
    #   online_order - If it was an online order.
    #   fraud - If the transaction is fraudulent.
    Data = pd.read_csv(data_input_path)

    # Set the input (X) and output (Y) data.
    # The only output data we have is if it's fraudulent or not, and all other fields go as inputs to the model.

    X = Data.drop(columns = ['repeat_retailer','distance_from_home', 'fraud'])
    y = Data['fraud']

    # Split the data into training and testing sets so we have something to test the trained model with.

    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle = False)

    X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, stratify = y_train)

    # Scale the data to remove mean and have unit variance. This means that the data will be between -1 and 1, which makes it a lot easier for the model to learn than random potentially large values.
    # It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train.values)

    Path("artifact").mkdir(parents=True, exist_ok=True)
    with open("artifact/test_data.pkl", "wb") as handle:
        pickle.dump((X_test, y_test), handle)
    with open("artifact/scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)

    # Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), we set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.

    class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
    class_weights = {i : class_weights[i] for i in range(len(class_weights))}


    # Build the model, the model we build here is a simple fully connected deep neural network, containing 3 hidden layers and one output layer.

    model = Sequential()
    model.add(Dense(32, activation = 'relu', input_dim = len(X.columns)))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()


    # Train the model and get performance

    epochs = 2
    history = model.fit(X_train, y_train, epochs=epochs, \
                        validation_data=(scaler.transform(X_val.values),y_val), \
                        verbose = True, class_weight = class_weights)

    # Save the model as ONNX for easy use of ModelMesh

    model_proto, _ = tf2onnx.convert.from_keras(model)
    print(model_output_path)
    onnx.save(model_proto, model_output_path)


@dsl.component(
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023b-20240301",
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
    csv_file = get_data_task.outputs["data_output_path"]
    # csv_file = get_data_task.output
    train_model_task = train_model(data_input_path=csv_file)
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
