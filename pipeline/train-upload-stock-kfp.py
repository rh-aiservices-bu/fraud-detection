import kfp
from kfp.components import create_component_from_func

from kubernetes.client.models import *
import os

from get_data import get_data
from train_model import train_model
from upload import upload


get_data_component = create_component_from_func(
    get_data,
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023a-20230509-869b370",
    packages_to_install=["pandas-datareader", "yfinance"]
)

train_model_component = create_component_from_func(
    train_model,
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023a-20230509-869b370",
    packages_to_install=[
        "flatbuffers<3.0,>=1.12",
        "numpy==1.23.*",
        "pandas==2.0.*",
        "pandas-datareader==0.10.*",
        "scikit-learn==1.3.*",
        "tensorflow==2.11.*",
        "tf2onnx==1.14.*",
        "yfinance==0.2.23"
    ]
)

upload_model_component = create_component_from_func(
    upload,
    base_image="quay.io/modh/runtime-images:runtime-cuda-tensorflow-ubi9-python-3.9-2023a-20230509-869b370",
    packages_to_install=["boto3", "botocore"]
)


@kfp.dsl.pipeline(name="train_upload_stock_kfp")
def sdk_pipeline():
    get_data_task = get_data_component()
    csv_file = get_data_task.output
    train_model_task = train_model_component(csv_file)
    onnx_file = train_model_task.output
    upload_model_task = upload_model_component(onnx_file)

    upload_model_task.add_env_variable(V1EnvVar(
        name="S3_KEY",
        value="models/stocks.onnx"))

    upload_model_task.container.add_env_from(
        V1EnvFromSource(
            secret_ref=V1SecretReference(
                name="aws-connection-my-storage"
            )
        )
    )


from kfp_tekton.compiler import TektonCompiler


# DEFAULT_STORAGE_CLASS needs to be filled out with the correct storage class, or else it will default to kfp-csi-s3
os.environ["DEFAULT_STORAGE_CLASS"] = os.environ.get(
    "DEFAULT_STORAGE_CLASS", "gp3"
)
os.environ["DEFAULT_ACCESSMODES"] = os.environ.get(
    "DEFAULT_ACCESSMODES", "ReadWriteOnce"
)
TektonCompiler().compile(sdk_pipeline, __file__.replace(".py", ".yaml"))
