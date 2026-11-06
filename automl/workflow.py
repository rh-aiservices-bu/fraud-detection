"""Small shared helpers for the AutoML workshop notebooks; no AutoX dependency."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
import warnings
from contextlib import contextmanager, redirect_stdout
from datetime import datetime, timezone
from functools import partial
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import quote, urlsplit

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
FEATURES = [
    "distance_from_home", "distance_from_last_transaction",
    "ratio_to_median_purchase_price", "repeat_retailer", "used_chip",
    "used_pin_number", "online_order",
]
LABEL = "fraud"


def settings():
    """Read current .env values without caching them in the kernel environment."""
    return {**dotenv_values(ROOT / ".env"), **os.environ}


def required(name):
    value = settings().get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} in automl/.env or the workbench environment.")
    return value


def save(name, value):
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    return path


def read(name):
    path = OUTPUTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing {path.name}; complete the preceding notebook first.")
    return json.loads(path.read_text())


def s3_client():
    import boto3

    env = settings()
    endpoint = required("AWS_S3_ENDPOINT")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or "YOUR_" in endpoint:
        source = ("the workbench/kernel environment" if "AWS_S3_ENDPOINT" in os.environ
                  else str(ROOT / ".env"))
        raise ValueError(
            f"AWS_S3_ENDPOINT from {source} must be a real http(s) S3 endpoint. "
            "Replace example placeholders. Workbench environment variables override .env."
        )
    return boto3.client(
        "s3", endpoint_url=endpoint,
        aws_access_key_id=env.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY") or None,
        aws_session_token=env.get("AWS_SESSION_TOKEN") or None,
        region_name=env.get("AWS_DEFAULT_REGION", "us-east-1"),
        verify=env.get("AWS_CA_BUNDLE") or True,
    )


def kfp_client():
    import kfp

    env = settings()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^This client only works with Kubeflow Pipeline v2\.0\.0-beta\.2 and later versions\.$",
            category=FutureWarning,
            module=r"^kfp\.client\.client$",
        )
        return kfp.Client(
            host=required("KFP_ENDPOINT"), namespace=required("RHOAI_PROJECT"),
            existing_token=required("RHOAI_TOKEN"),
            ssl_ca_cert=env.get("REQUESTS_CA_BUNDLE") or None,
        )


def data_path(value):
    """Resolve local CSV paths relative to automl/, independently of notebook cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def dataset_paths():
    """Default source files; an empty VALIDATION_CSV disables that optional split."""
    env = settings()
    return {"train": env.get("TRAIN_CSV", "../data/train.csv"),
            "test": env.get("TEST_CSV", "../data/test.csv"),
            "validate": env.get("VALIDATION_CSV", "../data/validate.csv")}


def merge_validation_enabled(value=None):
    value = settings().get("MERGE_VALIDATION", "true") if value is None else value
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError("MERGE_VALIDATION must be true or false")


def load_data(paths=None):
    """Load chosen local CSVs with the seven fraud features and binary label."""
    import numpy as np
    import pandas as pd

    paths = dataset_paths() if paths is None else paths
    for split in ("train", "test"):
        if not paths.get(split):
            raise ValueError(f"Choose a {split} CSV in notebook 00's dataset settings.")
    unknown = set(paths) - {"train", "validate", "test"}
    if unknown:
        raise ValueError(f"Unknown dataset roles: {sorted(unknown)}")
    resolved = {split: data_path(value) for split, value in paths.items() if value}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Train, test and optional validation must use different files; test must stay out of training.")
    frames = {}
    for split, path in resolved.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{split} CSV not found: {path}. Edit DATA_FILES in notebook 00 "
                "or TRAIN_CSV / TEST_CSV / VALIDATION_CSV in automl/.env. "
                "For no validation file, set VALIDATION_CSV to an empty value and MERGE_VALIDATION=false."
            )
        frame = pd.read_csv(path)
        if set(frame.columns) != set(FEATURES + [LABEL]):
            missing = sorted(set(FEATURES + [LABEL]) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(FEATURES + [LABEL]))
            raise ValueError(f"Unexpected columns in {split} CSV ({path}). Missing: {missing}; extra: {extra}")
        try:
            numeric = frame.to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"All features and fraud labels in {split} CSV ({path}) must be numeric.") from exc
        if frame.empty or not np.isfinite(numeric).all():
            raise ValueError(f"Empty, missing or non-finite values in {split} CSV ({path})")
        if set(frame[LABEL].unique()) != {0, 1}:
            raise ValueError(f"{split} CSV ({path}) must contain both binary fraud classes: 0 and 1")
        frame[LABEL] = frame[LABEL].astype(int)
        frames[split] = frame[FEATURES + [LABEL]]
        frames[split].attrs["source_path"] = str(path)
    return frames


def data_summary(frames):
    return [{"split": split, "rows": len(frame),
             "fraud_rows": int(frame[LABEL].sum()),
             "fraud_rate": float(frame[LABEL].mean())}
            for split, frame in frames.items()]


def prepared_datasets(frames, *, merge_validation=None):
    """Use the same explicit train/test selection for the preview and the upload."""
    import pandas as pd

    merge = merge_validation_enabled(merge_validation)
    if merge and "validate" not in frames:
        raise ValueError("MERGE_VALIDATION=True requires a validation CSV. Choose that file or set MERGE_VALIDATION=False.")
    train = (pd.concat([frames["train"], frames["validate"]], ignore_index=True)
             if merge else frames["train"])
    return {"train": train, "test": frames["test"]}


def stage_data(frames, s3, *, merge_validation=None):
    """Upload immutable per-experiment input copies and a manifest (no credentials)."""
    env = settings()
    merge = merge_validation_enabled(merge_validation)
    datasets = prepared_datasets(frames, merge_validation=merge)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    prefix = env.get("S3_DATA_PREFIX", "fraud-detection/automl/data").strip("/") + "/" + stamp
    bucket = required("AWS_S3_BUCKET")
    manifest = {"bucket": bucket, "connection": required("S3_CONNECTION_NAME"),
                "merged_validation": merge, "features": FEATURES,
                "source_paths": {split: frame.attrs.get("source_path") for split, frame in frames.items()},
                "source_splits": data_summary(frames), "datasets": {}}
    for split, frame in datasets.items():
        path = OUTPUTS / "data" / f"{split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        key = f"{prefix}/{split}.csv"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        s3.upload_file(str(path), bucket, key)
        head = s3.head_object(Bucket=bucket, Key=key)
        if head["ContentLength"] != path.stat().st_size:
            raise RuntimeError(f"Uploaded size mismatch: {key}")
        manifest["datasets"][split] = {
            "key": key, "uri": f"s3://{bucket}/{key}", "rows": len(frame),
            "fraud_rate": float(frame[LABEL].mean()), "sha256": digest,
        }
    save("data_manifest.json", manifest)
    s3.put_object(Bucket=bucket, Key=f"{prefix}/manifest.json",
                  Body=json.dumps(manifest, indent=2).encode(), ContentType="application/json")
    return manifest


def pipeline_arguments(manifest, *, preset=None, metric=None, top_n=None):
    env = settings()
    preset = env.get("AUTOML_PRESET", "speed") if preset is None else preset
    metric = env.get("AUTOML_EVAL_METRIC", "roc_auc") if metric is None else metric
    if preset not in {"speed", "balanced"}:
        raise ValueError("Use the GA speed or balanced preset for this workshop")
    if metric not in {"roc_auc", "f1", "accuracy"}:
        raise ValueError("Use roc_auc, f1 or accuracy as AUTOML_EVAL_METRIC")
    try:
        top_n = int(env.get("AUTOML_TOP_N", "1") if top_n is None else top_n)
    except (TypeError, ValueError) as exc:
        raise ValueError("AUTOML_TOP_N must be an integer between 1 and 10") from exc
    if not 1 <= top_n <= 10:
        raise ValueError("AUTOML_TOP_N must be between 1 and 10")
    return {
        "train_data_secret_name": manifest["connection"],
        "train_data_bucket_name": manifest["bucket"],
        "train_data_file_key": manifest["datasets"]["train"]["key"],
        "test_data_bucket_name": manifest["bucket"],
        "test_data_file_key": manifest["datasets"]["test"]["key"],
        "label_column": LABEL, "task_type": "binary", "positive_class": "1",
        "eval_metric": metric, "preset": preset, "top_n": top_n,
    }


def pipeline_input_names(spec):
    """Read KFP IR directly or inside a PipelineJob/API response wrapper."""
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Pipeline specification is not JSON; expected a KFP v2 specification.") from exc
    if hasattr(spec, "to_dict"):
        spec = spec.to_dict()
    if not isinstance(spec, dict):
        raise RuntimeError(f"Expected a pipeline specification mapping, got {type(spec).__name__}")
    top_keys = sorted(spec)
    # PipelineJob uses pipelineSpec; SDK model dictionaries may use snake_case.
    for _ in range(5):
        root = spec.get("root")
        if isinstance(root, dict):
            inputs = root.get("inputDefinitions", root.get("input_definitions", {}))
            parameters = inputs.get("parameters", {})
            if isinstance(parameters, dict):
                return set(parameters)
            break
        wrapper = next((spec[key] for key in ("pipelineSpec", "pipeline_spec", "spec")
                        if isinstance(spec.get(key), dict)), None)
        if wrapper is None:
            break
        spec = wrapper
    raise RuntimeError(
        "Cannot read pipeline inputs from the returned specification. "
        f"Top-level keys: {top_keys}; nested keys: {sorted(spec)}. "
        "Expected KFP v2 root.inputDefinitions.parameters, optionally inside pipelineSpec. "
        "No run was submitted. Share these key names to diagnose the server response format."
    )


def run_links(run):
    """Build RHOAI dashboard URLs using the recorded run's namespace and ID."""
    base = settings().get("RHOAI_DASHBOARD_URL", "").strip().rstrip("/")
    if not base:
        return {}
    parsed = urlsplit(base)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or "YOUR_" in base):
        raise ValueError("Set RHOAI_DASHBOARD_URL to the dashboard base URL, for example https://rh-ai.apps.YOUR_CLUSTER (with your actual cluster domain)")
    project = quote(run["project"], safe="")
    run_id = quote(run["run_id"], safe="")
    return {
        "Run details": f"{base}/develop-train/pipelines/runs/{project}/runs/{run_id}",
        "AutoML results": f"{base}/develop-train/automl/results/{project}/{run_id}",
    }


@contextmanager
def _without_sdk_links():
    """Suppress SDK-generated Kubeflow UI links; preserve stderr and exceptions."""
    try:
        from IPython.utils.capture import capture_output
    except ImportError:
        with redirect_stdout(StringIO()):
            yield
    else:
        with capture_output(stdout=True, stderr=False, display=True):
            yield


def submit_experiment(client, manifest, *, new_run=False, preset=None, metric=None,
                      top_n=None, experiment_name=None):
    """Resume the recorded run by default, avoiding accidental duplicate training."""
    if (OUTPUTS / "run.json").exists() and not new_run:
        saved = read("run.json")
        print(f"Reusing saved run {saved['run_id']}. To apply changed inputs or settings, set START_NEW_RUN=True for one execution.")
        return saved
    env = settings()
    args = pipeline_arguments(manifest, preset=preset, metric=metric, top_n=top_n)
    experiment_name = (env.get("EXPERIMENT_NAME", "fraud-detection-automl")
                       if experiment_name is None else experiment_name)
    if not experiment_name.strip():
        raise ValueError("EXPERIMENT_NAME must not be empty")
    name = env.get("MANAGED_PIPELINE_NAME", "autogluon-tabular-training-pipeline")
    pipeline_id = client.get_pipeline_id(name)
    if not pipeline_id:
        raise RuntimeError(f"Managed pipeline {name!r} not registered in this project; ask your administrator.")
    version_id = env.get("PIPELINE_VERSION_ID", "").strip()
    if not version_id:
        versions = client.list_pipeline_versions(
            pipeline_id=pipeline_id, page_size=1, sort_by="created_at desc"
        ).pipeline_versions
        if not versions:
            raise RuntimeError("Managed pipeline has no version")
        version_id = versions[0].pipeline_version_id
    # Fail before launching if this installed version lacks external holdout/preset inputs.
    version = client.get_pipeline_version(pipeline_id=pipeline_id, pipeline_version_id=version_id)
    definitions = pipeline_input_names(version.pipeline_spec)
    missing = set(args) - definitions
    if missing:
        raise RuntimeError(f"Managed pipeline version lacks {sorted(missing)}. Update the installed AutoML pipeline.")
    with _without_sdk_links():
        experiment = client.create_experiment(
            name=experiment_name,
            namespace=required("RHOAI_PROJECT"),
        )
    job_name = "fraud-automl-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    with _without_sdk_links():
        run = client.run_pipeline(
            experiment_id=experiment.experiment_id, job_name=job_name,
            pipeline_id=pipeline_id, version_id=version_id, params=args, enable_caching=False,
        )
    record = {"run_id": run.run_id, "pipeline_id": pipeline_id,
              "pipeline_version_id": version_id, "project": required("RHOAI_PROJECT"),
              "experiment_name": experiment_name,
              "arguments": args, "data": manifest, "state": "SUBMITTED",
              "submitted_at": datetime.now(timezone.utc).isoformat(),
              "artifact_bucket": env.get("S3_ARTIFACTS_BUCKET") or manifest["bucket"],
              "artifact_prefix": env.get("S3_ARTIFACTS_PREFIX", name + "/{run_id}").format(run_id=run.run_id).strip("/")}
    save("run.json", record)
    return record


def monitor_experiment(client=None):
    """Poll with progress; persist terminal state and server-measured duration."""
    from kfp_server_api.exceptions import ApiException

    record = read("run.json")
    deadline = time.monotonic() + int(settings().get("PIPELINE_TIMEOUT_SECONDS", "10800"))
    while True:
        try:
            if client is None:
                client = kfp_client()
            detail = client.get_run(record["run_id"])
        except ApiException as exc:
            if exc.status == 401:
                raise RuntimeError(
                    f"Pipeline API returned HTTP 401 while monitoring run {record['run_id']}. "
                    "RHOAI_TOKEN may be expired or invalid. Renew oc login if needed, "
                    "obtain the current token with oc whoami -t, and update RHOAI_TOKEN in automl/.env. "
                    "A workbench/kernel RHOAI_TOKEN overrides the file; update that value or restart the kernel if needed. "
                    "Then rerun wf.monitor_experiment() to read current credentials and monitor the same run. "
                    "This monitoring error did not cancel the run; no new submission or upload is needed."
                ) from None
            if exc.status == 403:
                raise RuntimeError(
                    f"Pipeline API returned HTTP 403 while monitoring run {record['run_id']}. "
                    "Check that the identity used by RHOAI_TOKEN can read runs in the saved project's pipeline server. "
                    "After correcting access, rerun wf.monitor_experiment(). The run was not canceled."
                ) from None
            raise
        run = getattr(detail, "run", detail)
        state = getattr(run, "state", None)
        state = str(getattr(state, "name", state)).upper()
        record["state"] = state
        start, end = getattr(run, "scheduled_at", None), getattr(run, "finished_at", None)
        if start and end:
            record["duration_seconds"] = (end - start).total_seconds()
        save("run.json", record)
        print(f"{datetime.now(timezone.utc):%H:%M:%S} {record['run_id']}: {state}", flush=True)
        if state in {"SUCCEEDED", "FAILED", "ERROR", "SYSTEM_ERROR", "CANCELED", "CANCELLED", "SKIPPED"}:
            if state != "SUCCEEDED":
                raise RuntimeError(f"Run ended with {state}. Inspect failed task logs for OOMKilled; record evidence, then rerun explicitly.")
            return record
        if time.monotonic() >= deadline:
            raise TimeoutError("Stopped waiting; the pipeline is still running. Resume this cell; do not submit a duplicate.")
        time.sleep(30)


def list_objects(s3, bucket, prefix):
    return [obj for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix.rstrip("/") + "/") for obj in page.get("Contents", [])]


def collect_results(s3):
    """Read per-model holdout metrics and download the official generated notebooks."""
    import nbformat
    import pandas as pd

    run = read("run.json")
    if run["state"] != "SUCCEEDED":
        raise RuntimeError("Complete monitoring of a successful run before collecting results")
    bucket, prefix = run["artifact_bucket"], run["artifact_prefix"]
    objects = list_objects(s3, bucket, prefix)
    keys = [obj["Key"] for obj in objects]
    if not keys:
        raise RuntimeError(f"No artifacts under s3://{bucket}/{prefix}. Check run output URIs and S3_ARTIFACTS_PREFIX; update run.json if needed.")
    save("artifact_index.json", [{"Key": obj["Key"], "Size": obj.get("Size", 0)} for obj in objects])
    rows = []
    for key in keys:
        if key.endswith("/metrics/metrics.json"):
            model_prefix = key.removesuffix("/metrics/metrics.json")
            metrics = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            rows.append({"model": model_prefix.rsplit("/", 1)[-1], **metrics,
                         "predictor_prefix": model_prefix + "/predictor/",
                         "notebook_key": model_prefix + "/notebooks/automl_predictor_notebook.ipynb"})
    metric = run["arguments"]["eval_metric"]
    if not rows or any(metric not in row or not math.isfinite(float(row[metric])) for row in rows):
        raise RuntimeError(f"Expected finite {metric} in per-model metrics; inspect artifact_index.json")
    rows.sort(key=lambda row: float(row[metric]), reverse=True)
    best = rows[0]
    if best["predictor_prefix"] + "predictor.pkl" not in keys:
        raise RuntimeError("Selected predictor directory is incomplete (missing predictor.pkl)")
    # Verify external test data was actually used, rather than an internal default split.
    test_keys = [key for key in keys if key.rsplit("/", 1)[-1] in
                 {"sampled_test_dataset.parquet", "sampled_test_dataset.csv"}]
    if len(test_keys) != 1:
        raise RuntimeError(f"Expected one external test artifact, found {test_keys}")
    test_bytes = s3.get_object(Bucket=bucket, Key=test_keys[0])["Body"].read()
    test = (pd.read_parquet(BytesIO(test_bytes)) if test_keys[0].endswith(".parquet")
            else pd.read_csv(BytesIO(test_bytes)))
    # Compare with the exact uploaded file recorded for this run, not today's
    # local sample file or a new path chosen after training.
    source_dataset = run["data"]["datasets"]["test"]
    source_bytes = s3.get_object(
        Bucket=run["data"]["bucket"], Key=source_dataset["key"])["Body"].read()
    if hashlib.sha256(source_bytes).hexdigest() != source_dataset["sha256"]:
        raise RuntimeError("The uploaded test CSV changed after staging. Use the original run's immutable inputs or submit a new run.")
    source = pd.read_csv(BytesIO(source_bytes))
    pd.testing.assert_frame_equal(
        test[FEATURES + [LABEL]].reset_index(drop=True),
        source[FEATURES + [LABEL]].reset_index(drop=True), check_dtype=False,
    )
    downloads = {}
    for filename, candidates in {
        "automl_experiment_notebook.ipynb": [k for k in keys if k.endswith("/automl_experiment_notebook.ipynb")],
        "automl_predictor_notebook.ipynb": [best["notebook_key"]] if best["notebook_key"] in keys else [],
    }.items():
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one official {filename}; found {candidates}")
        path = OUTPUTS / "generated" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, candidates[0], str(path))
        nbformat.validate(nbformat.read(path, as_version=4))
        downloads[filename] = str(path.relative_to(ROOT))
    html_keys = [key for key in keys if "html_artifact" in key]
    if html_keys:
        s3.download_file(bucket, html_keys[0], str(OUTPUTS / "leaderboard.html"))
    result = {"run_id": run["run_id"], "metric": metric, "holdout_score": best[metric],
              "selected_model": best["model"], "predictor_prefix": best["predictor_prefix"],
              "predictor_uri": f"s3://{bucket}/{best['predictor_prefix']}",
              "holdout_rows": len(test), "models": rows, "notebooks": downloads,
              "leaderboard_html": "outputs/leaderboard.html" if html_keys else None}
    save("results.json", result)
    return result


def kubernetes_api():
    from kubernetes import client, config
    from kubernetes.config.config_exception import ConfigException

    try:
        config.load_incluster_config()
    except ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi()


def runtime_template(api):
    """Fetch the configured installed template, with actionable discovery on 404."""
    from kubernetes.client.rest import ApiException

    env = settings()
    namespace = env.get("RUNTIME_TEMPLATE_NAMESPACE", "redhat-ods-applications")
    name = env.get("RUNTIME_TEMPLATE_NAME", "autogluon-runtime-template")
    try:
        return api.get_namespaced_custom_object(
            "template.openshift.io", "v1", namespace, "templates", name,
        )
    except ApiException as exc:
        if exc.status != 404:
            raise
        try:
            templates = api.list_namespaced_custom_object(
                "template.openshift.io", "v1", namespace, "templates",
            )["items"]
        except ApiException as listing_error:
            raise RuntimeError(
                f"Runtime template {name!r} was not found in {namespace!r}; "
                f"listing templates also failed (HTTP {listing_error.status}). "
                "Ask your administrator for the installed AutoGluon template name and namespace. "
                "Set RUNTIME_TEMPLATE_NAME and RUNTIME_TEMPLATE_NAMESPACE accordingly."
            ) from exc
        available = sorted(template["metadata"]["name"] for template in templates)
        candidates = sorted(template["metadata"]["name"] for template in templates
                            if any(obj.get("kind") == "ServingRuntime" and
                                   any(fmt.get("name", "").lower() == "autogluon"
                                       for fmt in obj.get("spec", {}).get("supportedModelFormats", []))
                                   for obj in template.get("objects", [])))
        raise RuntimeError(
            f"Runtime template {name!r} was not found in {namespace!r}. "
            f"AutoGluon template candidates: {candidates}. Available templates: {available}. "
            "Set RUNTIME_TEMPLATE_NAME to the installed AutoGluon template's metadata.name "
            "and check RUNTIME_TEMPLATE_NAMESPACE, then rerun manifest generation. "
            "The server image name need not equal the template name. "
            "If no AutoGluon template is installed, ask your administrator to enable the compatible runtime."
        ) from exc


def deployment_manifests(api):
    """Use the cluster's supported runtime image, never a hardcoded upstream image."""
    env = settings()
    run, result = read("run.json"), read("results.json")
    if result["run_id"] != run["run_id"]:
        raise ValueError("Collect results for the current run before deploying")
    namespace = required("RHOAI_PROJECT")
    if namespace != run["project"]:
        raise ValueError("Serving project must match the recorded pipeline project")
    template = runtime_template(api)
    embedded = next((obj for obj in template.get("objects", []) if obj.get("kind") == "ServingRuntime"), None)
    if embedded is None:
        raise ValueError("The configured template contains no ServingRuntime; select the installed AutoGluon runtime template")
    runtime_name = env.get("SERVING_RUNTIME_NAME", "fraud-automl-runtime")
    annotations = {k: v for k, v in embedded.get("metadata", {}).get("annotations", {}).items()
                   if k in {"opendatahub.io/apiProtocol", "opendatahub.io/runtime-version",
                            "opendatahub.io/kserve-runtime", "openshift.io/display-name"}}
    runtime = {"apiVersion": "serving.kserve.io/v1alpha1", "kind": "ServingRuntime",
               "metadata": {"name": runtime_name, "namespace": namespace,
                            "labels": {"opendatahub.io/dashboard": "true"}, "annotations": annotations},
               "spec": embedded["spec"]}
    if "${" in json.dumps(runtime):
        raise ValueError("Runtime template requires parameter substitution; deploy it through the dashboard instead")
    connection = run["data"]["connection"]
    account_name = env.get("MODEL_NAME", "fraud-automl") + "-storage"
    account = {"apiVersion": "v1", "kind": "ServiceAccount",
               "metadata": {"name": account_name, "namespace": namespace},
               "secrets": [{"name": connection}], "automountServiceAccountToken": False}
    role = {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role",
            "metadata": {"name": account_name, "namespace": namespace},
            "rules": [{"apiGroups": [""], "resources": ["secrets"],
                       "resourceNames": [connection], "verbs": ["get"]}]}
    binding = {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "RoleBinding",
               "metadata": {"name": account_name, "namespace": namespace},
               "subjects": [{"kind": "ServiceAccount", "name": account_name, "namespace": namespace}],
               "roleRef": {"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": account_name}}
    # Model-controller resolves this existing RHOAI Connection into storage credentials.
    service = {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
               "metadata": {"name": env.get("MODEL_NAME", "fraud-automl"), "namespace": namespace,
                            "labels": {"opendatahub.io/dashboard": "true", "networking.kserve.io/visibility": "exposed"},
                            "annotations": {
                                "serving.kserve.io/deploymentMode": "RawDeployment",
                                "security.opendatahub.io/enable-auth": "true",
                                "opendatahub.io/connections": connection,
                                "opendatahub.io/connection-path": result["predictor_prefix"].rstrip("/"),
                                "opendatahub.io/model-type": "predictive"}},
               "spec": {"predictor": {
                   "serviceAccountName": account_name, "automountServiceAccountToken": False,
                   "minReplicas": 1, "maxReplicas": 1,
                   "model": {"runtime": runtime_name, "modelFormat": {"name": "autogluon", "version": "1"},
                             "storage": {"key": connection, "path": result["predictor_prefix"].rstrip("/")},
                             "resources": {kind: {"cpu": env.get("SERVING_CPU", "2"),
                                                   "memory": env.get("SERVING_MEMORY", "8Gi")}
                                           for kind in ("requests", "limits")}}}}}
    # The serving Connection must point at the artifact bucket, not just the input bucket.
    from kubernetes import client
    import base64

    secret = client.CoreV1Api(api.api_client).read_namespaced_secret(connection, namespace)
    secret_bucket = base64.b64decode(secret.data.get("AWS_S3_BUCKET", "")).decode()
    if secret_bucket != run["artifact_bucket"]:
        raise ValueError("The serving Connection bucket must equal S3_ARTIFACTS_BUCKET. Create a Connection for that bucket and deploy via the dashboard.")
    manifests = [account, role, binding, runtime, service]
    save("deployment.json", manifests)
    return manifests


def manifest_differences(expected, actual, path=""):
    """Compare requested fields, allowing API-defaulted fields and metadata."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [path]
        return [difference for key, value in expected.items()
                for difference in manifest_differences(value, actual.get(key), f"{path}.{key}".lstrip("."))]
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [path]
        # Rules and subjects must match exactly, including any additional grants.
        if path in {"rules", "subjects"}:
            return [] if expected == actual else [path]
        if path != "secrets" and len(expected) != len(actual):
            return [path]
        differences = []
        for index, value in enumerate(expected):
            if isinstance(value, dict) and "name" in value:
                found = next((item for item in actual if isinstance(item, dict)
                              and item.get("name") == value["name"]), None)
            else:
                found = actual[index] if index < len(actual) else None
            differences.extend(manifest_differences(value, found, f"{path}[{index}]"))
        return differences
    return [] if expected == actual else [path]


def create_deployment(api, manifests):
    """Create missing resources and reuse matching ones without overwriting them."""
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    core = client.CoreV1Api(api.api_client)
    rbac = client.RbacAuthorizationV1Api(api.api_client)
    serializer = client.ApiClient()
    outcomes = []
    for obj in manifests:
        namespace = obj["metadata"]["namespace"]
        name = obj["metadata"]["name"]
        kind = obj["kind"]
        if kind == "ServiceAccount":
            create = partial(core.create_namespaced_service_account, namespace, obj)
            get = partial(core.read_namespaced_service_account, name, namespace)
        elif kind == "Role":
            create = partial(rbac.create_namespaced_role, namespace, obj)
            get = partial(rbac.read_namespaced_role, name, namespace)
        elif kind == "RoleBinding":
            create = partial(rbac.create_namespaced_role_binding, namespace, obj)
            get = partial(rbac.read_namespaced_role_binding, name, namespace)
        else:
            version = obj["apiVersion"].split("/")[1]
            plural = {"ServingRuntime": "servingruntimes", "InferenceService": "inferenceservices"}[kind]
            create = partial(api.create_namespaced_custom_object, "serving.kserve.io", version, namespace, plural, obj)
            get = partial(api.get_namespaced_custom_object, "serving.kserve.io", version, namespace, plural, name)
        try:
            create()
            action = "created"
        except ApiException as exc:
            if exc.status != 409:
                raise
            existing = serializer.sanitize_for_serialization(get())
            differences = manifest_differences(obj, existing)
            if differences:
                raise RuntimeError(
                    f"{kind} {namespace}/{name} already exists with different configuration. "
                    f"Differing fields: {', '.join(differences)}. Nothing was overwritten. "
                    "Review the existing resource in the dashboard or select a different "
                    "MODEL_NAME/SERVING_RUNTIME_NAME and regenerate manifests."
                ) from exc
            action = "reused"
        outcomes.append({"kind": kind, "name": name, "action": action})
        print(f"{kind} {namespace}/{name}: {action}", flush=True)
    return outcomes


def serving_endpoint(api):
    env = settings()
    namespace, name = required("RHOAI_PROJECT"), env.get("MODEL_NAME", "fraud-automl")
    deadline = time.monotonic() + int(env.get("SERVING_TIMEOUT_SECONDS", "1200"))
    while time.monotonic() < deadline:
        service = api.get_namespaced_custom_object(
            "serving.kserve.io", "v1beta1", namespace, "inferenceservices", name)
        status = service.get("status", {})
        conditions = status.get("conditions", [])
        print([(c["type"], c["status"], c.get("message", "")) for c in conditions], flush=True)
        if any(c["type"] == "Ready" and c["status"] == "True" for c in conditions):
            if env.get("PREDICT_ENDPOINT"):
                return env["PREDICT_ENDPOINT"].rstrip("/")
            routes = api.list_namespaced_custom_object("route.openshift.io", "v1", namespace, "routes")["items"]
            for route in routes:
                if route["metadata"]["name"] in {name, name + "-predictor"}:
                    scheme = "https" if route["spec"].get("tls") else "http"
                    return f"{scheme}://{route['spec']['host']}"
            if status.get("url"):
                return status["url"].rstrip("/")
            raise RuntimeError("Ready, but no endpoint; set PREDICT_ENDPOINT from the dashboard")
        time.sleep(10)
    raise TimeoutError("Serving readiness timed out. Inspect model events, storage credentials and pod logs in the dashboard.")


def prediction_headers(env):
    """Resolve inference credentials per request; never persist or display tokens."""
    mode = env.get("PREDICT_AUTH_MODE", "auto").strip().lower()
    if mode not in {"auto", "token", "none"}:
        raise ValueError("PREDICT_AUTH_MODE must be auto, token or none")
    if mode == "none":
        return {}
    token = env.get("PREDICT_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    if mode == "token":
        raise ValueError("PREDICT_AUTH_MODE=token requires PREDICT_TOKEN")

    from kubernetes import client, config
    from kubernetes.config.config_exception import ConfigException

    configuration = client.Configuration()
    try:
        try:
            config.load_incluster_config(client_configuration=configuration)
        except ConfigException:
            config.load_kube_config(client_configuration=configuration)
        # The SDK refreshes credentials here when its authentication provider supports it.
        authorization = configuration.get_api_key_with_prefix("authorization")
    except ConfigException:
        authorization = None
    if not authorization or not authorization.lower().startswith("bearer "):
        raise ValueError(
            "No inference bearer token available. Run oc login for the serving cluster, "
            "use an authorized workbench ServiceAccount, or set PREDICT_TOKEN. "
            "For an intentionally public endpoint set PREDICT_AUTH_MODE=none."
        )
    return {"Authorization": authorization}


def prediction_error_detail(response, env, headers):
    """Bound error output and redact credentials known to this request."""
    detail = response.text
    secrets = [env.get(key, "") for key in (
        "PREDICT_TOKEN", "RHOAI_TOKEN", "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    )]
    authorization = headers.get("Authorization", "")
    if authorization:
        secrets.extend([authorization, authorization.split(" ", 1)[-1]])
    for secret in sorted({value for value in secrets if value}, key=len, reverse=True):
        detail = detail.replace(secret, "[REDACTED]")
    detail = re.sub(r"(?i)\bbearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", detail)
    return detail[:2000] + (" ... [truncated]" if len(detail) > 2000 else "")


def predict(endpoint, rows):
    """Accept scalar records; encode list-valued instances for AutoGluon KServe v1."""
    import requests

    if not rows or any(set(row) != set(FEATURES) for row in rows):
        raise ValueError("Each instance must include exactly the seven training features")
    env = settings()
    name = env.get("MODEL_NAME", "fraud-automl")
    if read("results.json")["run_id"] != read("run.json")["run_id"]:
        raise ValueError("Collect results for the current run before scoring")
    # An unsuccessful retry must not leave a previous smoke-test success in the report.
    save("serving_result.json", {"run_id": read("run.json")["run_id"],
                                 "serving_predict_worked": False})
    headers = prediction_headers(env)
    # KServe get_predict_input builds a DataFrame from each instance. Each feature
    # must therefore be a column list, even when that instance contains one row.
    instances = [{feature: [row[feature]] for feature in FEATURES} for row in rows]
    response = requests.post(
        f"{endpoint.rstrip('/')}/v1/models/{name}:predict", json={"instances": instances},
        headers=headers, timeout=120, verify=env.get("REQUESTS_CA_BUNDLE") or True,
    )
    if response.status_code in {401, 403}:
        raise requests.HTTPError(
            f"Inference returned HTTP {response.status_code}. "
            "For automatic credentials, renew oc login or check the workbench identity. "
            "A configured PREDICT_TOKEN overrides automatic credentials; clear it to use auto mode. "
            "Verify that the identity has model inference access. "
            "If the endpoint is intentionally public, use PREDICT_AUTH_MODE=none.",
            response=response,
        )
    if response.status_code >= 400:
        detail = prediction_error_detail(response, env, headers)
        save("serving_result.json", {"run_id": read("run.json")["run_id"],
                                     "serving_predict_worked": False,
                                     "http_status": response.status_code})
        raise requests.HTTPError(
            f"Inference returned HTTP {response.status_code}. "
            f"Server response: {detail or '(empty)'}. "
            "Inspect the kserve-container pod logs for this InferenceService; "
            "the status alone does not identify the runtime/model error.",
            response=response,
        )
    response.raise_for_status()
    result = response.json()
    values = result.get("predictions")
    if not isinstance(values, list) or len(values) != len(rows):
        raise ValueError(f"Expected one prediction per instance; received {result}")
    save("serving_result.json", {"run_id": read("run.json")["run_id"], "model_name": name,
                                 "endpoint": endpoint, "serving_predict_worked": True,
                                 "instances": rows, "response": result})
    return result


def ticket_report():
    run = read("run.json")
    results = read("results.json") if (OUTPUTS / "results.json").exists() else {}
    serving = read("serving_result.json") if (OUTPUTS / "serving_result.json").exists() else {}
    # A new run must not inherit evidence from a previous run.
    if results.get("run_id") != run["run_id"]:
        results = {}
    if serving.get("run_id") != run["run_id"]:
        serving = {}
    lines = ["AutoML additional workflow (RHOAIENG-96834)",
             "Branch: <fill in branch URL>", "PR: <fill in PR URL>"]
    lines.extend(f"{label}: {url}" for label, url in run_links(run).items())
    for split, data in run["data"]["datasets"].items():
        lines.append(f"{split}: {data['uri']}; rows={data['rows']}; fraud rate={data['fraud_rate']:.6%}")
    lines += [f"Run: {run['run_id']}; version: {run['pipeline_version_id']}",
              f"Preset: {run['arguments']['preset']}; state: {run['state']}",
              f"Duration (seconds): {run.get('duration_seconds', 'pending')}",
              "OOM: <inspect failed pod termination reason; do not infer OOM from FAILED>",
              f"Metric: {run['arguments']['eval_metric']}; external test score: {results.get('holdout_score', 'pending')}",
              f"Selected predictor: {results.get('predictor_uri', 'pending')}",
              f"Serving predict worked: {serving.get('serving_predict_worked', 'not tested')}"]
    report = "\n".join(lines) + "\n"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "ticket-comment.md").write_text(report)
    return report
