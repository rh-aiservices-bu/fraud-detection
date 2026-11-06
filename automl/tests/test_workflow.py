"""Offline contract checks. No cluster, credentials or AutoGluon installation needed."""

import ast
import base64
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nbformat
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import workflow as wf


class FakeS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key):
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[(bucket, key)])

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix):
        # Two pages ensure artifact discovery does not lose later models.
        items = [{"Key": key, "Size": len(value)}
                 for (bucket, key), value in self.objects.items()
                 if bucket == Bucket and key.startswith(Prefix)]
        yield {"Contents": items[:2]}
        yield {"Contents": items[2:]}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        for context in [
            patch.object(wf, "OUTPUTS", self.root / "automl" / "outputs"),
            patch.object(wf, "ROOT", self.root / "automl"),
            patch.dict(os.environ, {"AWS_S3_BUCKET": "bucket", "S3_CONNECTION_NAME": "connection",
                                    "RHOAI_PROJECT": "fraud", "MERGE_VALIDATION": "true",
                                    "PREDICT_AUTH_MODE": "none"}, clear=True),
        ]:
            context.start()
            self.addCleanup(context.stop)
        self.frames = {}
        for i, split in enumerate(("train", "validate", "test")):
            frame = pd.DataFrame([{feature: float(j + i + 1) for feature in wf.FEATURES} for j in range(4)])
            frame[wf.LABEL] = [0, 0, 0, 1]
            data = self.root / "data"
            data.mkdir(exist_ok=True)
            frame.to_csv(data / f"{split}.csv", index=False)
            self.frames[split] = frame
        self.s3 = FakeS3()

    def record_run(self, state="SUCCEEDED"):
        manifest = wf.stage_data(self.frames, self.s3)
        run = {"run_id": "run-123", "pipeline_version_id": "v1", "project": "fraud",
               "state": state, "arguments": wf.pipeline_arguments(manifest), "data": manifest,
               "artifact_bucket": "bucket", "artifact_prefix": "pipeline/run-123"}
        wf.save("run.json", run)
        return run

    def add_artifacts(self):
        run = self.record_run()
        prefix = run["artifact_prefix"]
        # Give lower ROC-AUC model higher accuracy to catch accidental accuracy ranking.
        for name, auc, accuracy in [("low", 0.7, 0.99), ("high", 0.98, 0.95)]:
            base = f"{prefix}/models/{name}"
            self.s3.put_object(Bucket="bucket", Key=f"{base}/metrics/metrics.json",
                               Body=json.dumps({"roc_auc": auc, "accuracy": accuracy}).encode())
            self.s3.put_object(Bucket="bucket", Key=f"{base}/predictor/predictor.pkl", Body=b"fake")
            self.s3.put_object(Bucket="bucket", Key=f"{base}/notebooks/automl_predictor_notebook.ipynb",
                               Body=nbformat.writes(nbformat.v4.new_notebook()).encode())
        self.s3.put_object(Bucket="bucket", Key=f"{prefix}/automl_experiment_notebook.ipynb",
                           Body=nbformat.writes(nbformat.v4.new_notebook()).encode())
        self.s3.put_object(Bucket="bucket", Key=f"{prefix}/sampled_test_dataset.csv",
                           Body=self.frames["test"].to_csv(index=False).encode())
        return run

    def test_staging_keeps_external_holdout_and_all_features(self):
        frames = wf.load_data()
        manifest = wf.stage_data(frames, self.s3)
        self.assertEqual(manifest["datasets"]["train"]["rows"], 8)
        self.assertEqual(manifest["datasets"]["test"]["rows"], 4)
        self.assertEqual(manifest["datasets"]["test"]["fraud_rate"], 0.25)
        staged = pd.read_csv(io.BytesIO(self.s3.objects[("bucket", manifest["datasets"]["train"]["key"])]))
        self.assertEqual(staged.columns.tolist(), wf.FEATURES + [wf.LABEL])
        self.assertNotEqual(manifest["datasets"]["train"]["key"], manifest["datasets"]["test"]["key"])
        again = wf.stage_data(frames, self.s3)
        self.assertNotEqual(manifest["datasets"]["train"]["key"], again["datasets"]["train"]["key"])

    def test_optional_validation_merge(self):
        with patch.dict(os.environ, {"MERGE_VALIDATION": "false"}):
            manifest = wf.stage_data(self.frames, self.s3)
        self.assertEqual(manifest["datasets"]["train"]["rows"], 4)
        self.assertEqual(manifest["datasets"]["test"]["rows"], 4)

    def test_custom_csv_paths_without_validation_and_explicit_merge_choice(self):
        custom = self.root / "custom"
        custom.mkdir()
        train_path, test_path = custom / "fit.csv", custom / "evaluate.csv"
        self.frames["train"].iloc[:3].assign(fraud=[0, 1, 0]).to_csv(train_path, index=False)
        self.frames["test"].iloc[:2].assign(fraud=[0, 1]).to_csv(test_path, index=False)
        frames = wf.load_data({"train": "../custom/fit.csv", "test": test_path, "validate": None})
        preview = wf.prepared_datasets(frames, merge_validation=False)
        manifest = wf.stage_data(frames, self.s3, merge_validation=False)
        self.assertEqual(set(frames), {"train", "test"})
        self.assertFalse(manifest["merged_validation"])
        self.assertEqual(manifest["source_paths"]["train"], str(train_path.resolve()))
        self.assertEqual(manifest["datasets"]["train"]["rows"], 3)
        self.assertEqual(manifest["datasets"]["test"]["rows"], 2)
        for role in ("train", "test"):
            uploaded = pd.read_csv(io.BytesIO(self.s3.objects[("bucket", manifest["datasets"][role]["key"])]))
            pd.testing.assert_frame_equal(uploaded, preview[role])

    def test_csv_paths_can_be_configured_in_environment(self):
        with patch.dict(os.environ, {"TRAIN_CSV": "../data/train.csv", "TEST_CSV": "../data/test.csv",
                                     "VALIDATION_CSV": "", "MERGE_VALIDATION": "false"}):
            frames = wf.load_data()
            manifest = wf.stage_data(frames, self.s3)
        self.assertEqual(set(frames), {"train", "test"})
        self.assertEqual(manifest["datasets"]["train"]["rows"], 4)

    def test_csv_selection_rejects_missing_files_and_shared_test_path(self):
        for paths, exception, message in [
            ({"train": "", "test": "../data/test.csv"}, ValueError, "Choose a train CSV"),
            ({"train": "../data/train.csv", "test": "../data/train.csv"}, ValueError, "different files"),
            ({"train": "../data/train.csv", "test": "missing.csv"}, FileNotFoundError, "test CSV not found"),
        ]:
            with self.subTest(paths=paths), self.assertRaisesRegex(exception, message):
                wf.load_data(paths)

    def test_missing_validation_fails_before_upload(self):
        with self.assertRaisesRegex(ValueError, "requires a validation CSV"):
            wf.stage_data({key: self.frames[key] for key in ("train", "test")}, self.s3)
        self.assertEqual(self.s3.objects, {})

    def test_explicit_experiment_options_override_env_without_mutation(self):
        manifest = wf.stage_data(self.frames, self.s3)
        args = wf.pipeline_arguments(manifest, preset="balanced", metric="f1", top_n=2)
        self.assertEqual((args["preset"], args["eval_metric"], args["top_n"]), ("balanced", "f1", 2))
        self.assertEqual(wf.pipeline_arguments(manifest)["preset"], "speed")
        with self.assertRaisesRegex(ValueError, "integer"):
            wf.pipeline_arguments(manifest, top_n="invalid")

    def test_holdout_verification_uses_uploaded_snapshot_not_current_local_file(self):
        self.add_artifacts()
        (self.root / "data/test.csv").unlink()
        with patch.dict(os.environ, {"TEST_CSV": "different.csv"}):
            result = wf.collect_results(self.s3)
        self.assertEqual(result["holdout_rows"], 4)

    def test_holdout_verification_rejects_modified_s3_source(self):
        run = self.add_artifacts()
        test = run["data"]["datasets"]["test"]
        self.s3.objects[("bucket", test["key"])] += b"\n"
        with self.assertRaisesRegex(RuntimeError, "changed after staging"):
            wf.collect_results(self.s3)
        self.assertFalse((wf.OUTPUTS / "results.json").exists())

    def test_env_edits_are_read_without_retaining_old_values(self):
        wf.ROOT.mkdir(parents=True, exist_ok=True)
        path = wf.ROOT / ".env"
        path.write_text("AWS_S3_ENDPOINT=https://s3.YOUR_REGION.amazonaws.com\n")
        self.assertIn("YOUR_REGION", wf.settings()["AWS_S3_ENDPOINT"])
        path.write_text("AWS_S3_ENDPOINT=https://s3.eu-central-1.amazonaws.com\n"
                        "AWS_ACCESS_KEY_ID=example-key\nAWS_SECRET_ACCESS_KEY=example-secret\n")
        with patch("boto3.client") as client:
            wf.s3_client()
        self.assertEqual(client.call_args.kwargs["endpoint_url"], "https://s3.eu-central-1.amazonaws.com")
        self.assertEqual(client.call_args.kwargs["aws_access_key_id"], "example-key")
        self.assertNotIn("AWS_S3_ENDPOINT", os.environ)
        with patch.dict(os.environ, {"AWS_S3_ENDPOINT": "https://workbench.example"}):
            self.assertEqual(wf.settings()["AWS_S3_ENDPOINT"], "https://workbench.example")

    def test_placeholder_error_identifies_overriding_environment(self):
        with patch.dict(os.environ, {"AWS_S3_ENDPOINT": "https://s3.YOUR_REGION.amazonaws.com"}):
            with patch("boto3.client") as client:
                with self.assertRaisesRegex(ValueError, "workbench/kernel environment"):
                    wf.s3_client()
                client.assert_not_called()

    def test_data_validation_rejects_missing_feature(self):
        self.frames["train"].drop(columns=[wf.FEATURES[0]]).to_csv(self.root / "data/train.csv", index=False)
        with self.assertRaisesRegex(ValueError, "Unexpected columns"):
            wf.load_data()

    def test_submit_contract_and_resume_without_duplicate(self):
        manifest = wf.stage_data(self.frames, self.s3)
        args = wf.pipeline_arguments(manifest)
        client = Mock()
        client.get_pipeline_id.return_value = "pipeline-id"
        client.list_pipeline_versions.return_value = SimpleNamespace(
            pipeline_versions=[SimpleNamespace(pipeline_version_id="version-id")])
        client.get_pipeline_version.return_value = SimpleNamespace(pipeline_spec={
            "root": {"inputDefinitions": {"parameters": {name: {} for name in args}}}})
        client.create_experiment.return_value = SimpleNamespace(experiment_id="experiment-id")
        client.run_pipeline.return_value = SimpleNamespace(run_id="new-run")
        record = wf.submit_experiment(client, manifest)
        self.assertEqual(record["arguments"]["preset"], "speed")
        self.assertEqual(record["arguments"]["eval_metric"], "roc_auc")
        self.assertEqual(record["arguments"]["test_data_file_key"], manifest["datasets"]["test"]["key"])
        self.assertEqual(client.run_pipeline.call_args.kwargs["version_id"], "version-id")
        wf.submit_experiment(client, manifest)
        client.run_pipeline.assert_called_once()

    def test_missing_external_test_inputs_fails_before_submission(self):
        manifest = wf.stage_data(self.frames, self.s3)
        client = Mock()
        client.list_pipeline_versions.return_value = SimpleNamespace(
            pipeline_versions=[SimpleNamespace(pipeline_version_id="version-id")])
        client.get_pipeline_version.return_value = SimpleNamespace(pipeline_spec={
            "root": {"inputDefinitions": {"parameters": {}}}})
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            wf.submit_experiment(client, manifest)
        client.run_pipeline.assert_not_called()

    def test_pipeline_inputs_accept_wrapped_and_serialized_specs(self):
        native = {"root": {"inputDefinitions": {"parameters": {
            "label_column": {}, "test_data_file_key": {}, "preset": {}}}}}
        for spec in [native, {"pipelineSpec": native, "platformSpec": {}},
                     {"pipeline_spec": native}, {"spec": {"pipelineSpec": native}},
                     json.dumps({"pipelineSpec": native}),
                     SimpleNamespace(to_dict=lambda: native)]:
            with self.subTest(spec=spec):
                self.assertEqual(wf.pipeline_input_names(spec),
                                 {"label_column", "test_data_file_key", "preset"})

    def test_unknown_pipeline_format_reports_keys(self):
        with self.assertRaisesRegex(RuntimeError, "Top-level keys:.*unknown"):
            wf.pipeline_input_names({"unknown": {}})

    def test_submission_accepts_pipeline_job_wrapper(self):
        manifest = wf.stage_data(self.frames, self.s3)
        args = wf.pipeline_arguments(manifest)
        client = Mock()
        client.get_pipeline_id.return_value = "pipeline-id"
        client.list_pipeline_versions.return_value = SimpleNamespace(
            pipeline_versions=[SimpleNamespace(pipeline_version_id="version-id")])
        client.get_pipeline_version.return_value = SimpleNamespace(pipeline_spec={
            "pipelineSpec": {"root": {"inputDefinitions": {
                "parameters": {name: {} for name in args}}}}, "platformSpec": {}})
        client.create_experiment.return_value = SimpleNamespace(experiment_id="experiment-id")
        client.run_pipeline.return_value = SimpleNamespace(run_id="wrapped-run")
        record = wf.submit_experiment(client, manifest, preset="balanced", metric="f1", top_n=2,
                                      experiment_name="custom-experiment")
        self.assertEqual(record["run_id"], "wrapped-run")
        self.assertEqual(client.run_pipeline.call_args.kwargs["params"],
                         {**args, "preset": "balanced", "eval_metric": "f1", "top_n": 2})
        self.assertEqual(record["experiment_name"], "custom-experiment")
        self.assertEqual(client.create_experiment.call_args.kwargs["name"], "custom-experiment")

    def test_dashboard_links_use_saved_project_and_rhoai_paths(self):
        base = "https://rh-ai.apps.example.com"
        run_id = "example-run-id"
        with patch.dict(os.environ, {"RHOAI_DASHBOARD_URL": base + "/", "RHOAI_PROJECT": "other"}):
            links = wf.run_links({"project": "your-project", "run_id": run_id})
        self.assertEqual(links["Run details"], f"{base}/develop-train/pipelines/runs/your-project/runs/{run_id}")
        self.assertEqual(links["AutoML results"], f"{base}/develop-train/automl/results/your-project/{run_id}")

    def test_dashboard_links_optional_and_validate_url(self):
        self.assertEqual(wf.run_links({"project": "fraud", "run_id": "id"}), {})
        with patch.dict(os.environ, {"RHOAI_DASHBOARD_URL": "https://rh-ai.apps.YOUR_CLUSTER"}):
            with self.assertRaisesRegex(ValueError, "RHOAI_DASHBOARD_URL"):
                wf.run_links({"project": "fraud", "run_id": "id"})
        with patch.dict(os.environ, {"RHOAI_DASHBOARD_URL": "https://dashboard.example"}):
            links = wf.run_links({"project": "name/with/slash", "run_id": "id?query"})
            self.assertIn("name%2Fwith%2Fslash", links["Run details"])
            self.assertIn("id%3Fquery", links["Run details"])

    def test_sdk_link_capture_preserves_errors_and_stderr(self):
        from contextlib import redirect_stdout, redirect_stderr

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                with wf._without_sdk_links():
                    print("Run details: incorrect-kubeflow-url")
                    print("SDK warning", file=sys.stderr)
                    raise RuntimeError("submission failed")
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("SDK warning", stderr.getvalue())

    def test_metric_selection_and_official_notebook_downloads(self):
        self.add_artifacts()
        result = wf.collect_results(self.s3)
        self.assertEqual(result["selected_model"], "high")
        self.assertEqual(result["holdout_score"], 0.98)
        self.assertEqual(result["holdout_rows"], 4)
        self.assertEqual(len(result["notebooks"]), 2)
        self.assertTrue((wf.OUTPUTS / "generated/automl_predictor_notebook.ipynb").exists())

    def test_wrong_holdout_artifact_is_rejected(self):
        self.add_artifacts()
        self.s3.put_object(Bucket="bucket", Key="pipeline/run-123/sampled_test_dataset.csv",
                           Body=self.frames["train"].to_csv(index=False).encode())
        with self.assertRaises(AssertionError):
            wf.collect_results(self.s3)

    def test_failed_run_is_recorded_without_oom_guess(self):
        self.record_run(state="RUNNING")
        client = Mock()
        client.get_run.return_value = SimpleNamespace(state="FAILED")
        with self.assertRaisesRegex(RuntimeError, "FAILED"):
            wf.monitor_experiment(client)
        self.assertEqual(wf.read("run.json")["state"], "FAILED")
        self.assertIn("external test score: pending", wf.ticket_report())

    def test_monitoring_recreates_client_and_resumes_saved_run(self):
        self.record_run(state="RUNNING")
        first, second = Mock(), Mock()
        for client in (first, second):
            client.get_run.return_value = SimpleNamespace(state="SUCCEEDED")
        with patch.object(wf, "kfp_client", side_effect=[first, second]) as factory:
            self.assertEqual(wf.monitor_experiment()["run_id"], "run-123")
            self.assertEqual(wf.monitor_experiment()["run_id"], "run-123")
        self.assertEqual(factory.call_count, 2)
        for client in (first, second):
            client.get_run.assert_called_once_with("run-123")
            client.run_pipeline.assert_not_called()

    def test_monitoring_auth_errors_preserve_state_and_hide_response(self):
        from kfp_server_api.exceptions import ApiException

        for status, message in [(401, "RHOAI_TOKEN may be expired"), (403, "can read runs")]:
            with self.subTest(status=status):
                self.record_run(state="RUNNING")
                before = (wf.OUTPUTS / "run.json").read_bytes()
                client = Mock()
                client.get_run.side_effect = ApiException(status=status, reason="private-response-marker")
                with self.assertRaisesRegex(RuntimeError, message) as caught:
                    wf.monitor_experiment(client)
                self.assertNotIn("private-response-marker", str(caught.exception))
                self.assertEqual((wf.OUTPUTS / "run.json").read_bytes(), before)
                client.run_pipeline.assert_not_called()

    def test_monitoring_handles_auth_error_during_client_creation(self):
        from kfp_server_api.exceptions import ApiException

        self.record_run(state="RUNNING")
        with patch.object(wf, "kfp_client", side_effect=ApiException(status=401)):
            with self.assertRaisesRegex(RuntimeError, "update RHOAI_TOKEN"):
                wf.monitor_experiment()

    def test_rest_named_features_and_failed_retry_clears_success(self):
        self.add_artifacts()
        wf.collect_results(self.s3)
        rows = self.frames["test"][wf.FEATURES].iloc[:2].to_dict(orient="records")
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"predictions": [0, 1]}
        with patch("requests.post", return_value=response) as post:
            wf.predict("https://model.example", rows)
        self.assertTrue(wf.read("serving_result.json")["serving_predict_worked"])
        encoded = post.call_args.kwargs["json"]["instances"]
        # Mirror the runtime's DataFrame-per-instance decoding. Scalar dictionaries
        # raise ValueError here; list-valued columns reconstruct the original rows.
        decoded = pd.concat([pd.DataFrame(instance) for instance in encoded], ignore_index=True)
        pd.testing.assert_frame_equal(decoded, pd.DataFrame(rows))
        self.assertTrue(all(len(value) == 1 for instance in encoded for value in instance.values()))
        self.assertEqual(post.call_args.args[0], "https://model.example/v1/models/fraud-automl:predict")
        self.assertTrue(post.call_args.kwargs["verify"])
        with patch("requests.post", side_effect=requests.ConnectionError("offline")):
            with self.assertRaises(requests.ConnectionError):
                wf.predict("https://model.example", rows)
        self.assertFalse(wf.read("serving_result.json")["serving_predict_worked"])

    def test_old_results_are_not_reported_for_new_run(self):
        self.record_run()
        wf.save("results.json", {"run_id": "other-run", "holdout_score": 1.0})
        wf.save("serving_result.json", {"run_id": "other-run", "serving_predict_worked": True})
        report = wf.ticket_report()
        self.assertIn("external test score: pending", report)
        self.assertIn("Serving predict worked: not tested", report)

    def test_prediction_auto_rereads_kubeconfig_credentials(self):
        from kubernetes.config.config_exception import ConfigException

        tokens = iter(["first-token", "refreshed-token"])

        def load_config(*, client_configuration):
            client_configuration.api_key["authorization"] = "Bearer " + next(tokens)

        with patch("kubernetes.config.load_incluster_config", side_effect=ConfigException("not in cluster")), \
             patch("kubernetes.config.load_kube_config", side_effect=load_config) as loader:
            self.assertEqual(wf.prediction_headers({})["Authorization"], "Bearer first-token")
            self.assertEqual(wf.prediction_headers({})["Authorization"], "Bearer refreshed-token")
        self.assertEqual(loader.call_count, 2)

    def test_prediction_auto_uses_workbench_credentials(self):
        def load_config(*, client_configuration):
            client_configuration.api_key["authorization"] = "bearer workbench-token"

        with patch("kubernetes.config.load_incluster_config", side_effect=load_config), \
             patch("kubernetes.config.load_kube_config") as kubeconfig:
            self.assertEqual(wf.prediction_headers({})["Authorization"], "bearer workbench-token")
        kubeconfig.assert_not_called()

    def test_prediction_explicit_token_and_public_mode(self):
        with patch("kubernetes.config.load_incluster_config") as loader:
            self.assertEqual(wf.prediction_headers({"PREDICT_TOKEN": "explicit"}),
                             {"Authorization": "Bearer explicit"})
            self.assertEqual(wf.prediction_headers({"PREDICT_AUTH_MODE": "none", "PREDICT_TOKEN": "stale"}), {})
            with self.assertRaisesRegex(ValueError, "requires PREDICT_TOKEN"):
                wf.prediction_headers({"PREDICT_AUTH_MODE": "token"})
            loader.assert_not_called()

    def test_prediction_missing_credentials_has_actionable_error(self):
        from kubernetes.config.config_exception import ConfigException

        with patch("kubernetes.config.load_incluster_config", side_effect=ConfigException("not in cluster")), \
             patch("kubernetes.config.load_kube_config", side_effect=ConfigException("no config")):
            with self.assertRaisesRegex(ValueError, "Run oc login"):
                wf.prediction_headers({})

    def test_prediction_unauthorized_error_and_evidence_do_not_include_token(self):
        self.add_artifacts()
        wf.collect_results(self.s3)
        rows = self.frames["test"][wf.FEATURES].iloc[:1].to_dict(orient="records")
        response = Mock(status_code=401)
        with patch.dict(os.environ, {"PREDICT_AUTH_MODE": "token", "PREDICT_TOKEN": "private-token"}), \
             patch("requests.post", return_value=response):
            with self.assertRaises(requests.HTTPError) as caught:
                wf.predict("https://model.example", rows)
        self.assertIn("renew oc login", str(caught.exception))
        self.assertNotIn("private-token", str(caught.exception))
        self.assertNotIn("private-token", (wf.OUTPUTS / "serving_result.json").read_text())

    def test_prediction_server_error_shows_redacted_body_and_records_failure(self):
        self.add_artifacts()
        wf.collect_results(self.s3)
        rows = self.frames["test"][wf.FEATURES].iloc[:1].to_dict(orient="records")
        response = requests.Response()
        response.status_code = 500
        response._content = b'{"error":"Model load failed; Bearer private-token; key example-secret"}'
        with patch.dict(os.environ, {"PREDICT_AUTH_MODE": "token", "PREDICT_TOKEN": "private-token",
                                     "AWS_SECRET_ACCESS_KEY": "example-secret"}), \
             patch("requests.post", return_value=response):
            with self.assertRaises(requests.HTTPError) as caught:
                wf.predict("https://model.example", rows)
        message = str(caught.exception)
        self.assertIn("Model load failed", message)
        self.assertNotIn("private-token", message)
        self.assertNotIn("example-secret", message)
        self.assertIs(caught.exception.response, response)
        self.assertEqual(wf.read("serving_result.json")["http_status"], 500)
        self.assertFalse(wf.read("serving_result.json")["serving_predict_worked"])

    def test_prediction_error_detail_is_bounded(self):
        response = requests.Response()
        response._content = b"x" * 5000
        detail = wf.prediction_error_detail(response, {}, {})
        self.assertLess(len(detail), 2050)
        self.assertIn("[truncated]", detail)

    def test_deployment_uses_installed_runtime_and_minimal_storage_access(self):
        self.add_artifacts()
        results = wf.collect_results(self.s3)
        api = Mock()
        api.get_namespaced_custom_object.return_value = {"objects": [{
            "kind": "ServingRuntime", "metadata": {},
            "spec": {"containers": [{"name": "kserve-container", "image": "registry.example/official:3.6"}]},
        }]}
        core = Mock()
        core.read_namespaced_secret.return_value = SimpleNamespace(
            data={"AWS_S3_BUCKET": base64.b64encode(b"bucket").decode()})
        with patch("kubernetes.client.CoreV1Api", return_value=core):
            manifests = wf.deployment_manifests(api)
        kinds = {obj["kind"]: obj for obj in manifests}
        self.assertEqual(kinds["ServingRuntime"]["spec"]["containers"][0]["image"], "registry.example/official:3.6")
        model = kinds["InferenceService"]["spec"]["predictor"]["model"]
        self.assertEqual(model["storage"]["path"], results["predictor_prefix"].rstrip("/"))
        self.assertEqual(kinds["Role"]["rules"], [{"apiGroups": [""], "resources": ["secrets"],
                                                   "resourceNames": ["connection"], "verbs": ["get"]}])
        self.assertEqual(kinds["InferenceService"]["metadata"]["annotations"]["security.opendatahub.io/enable-auth"], "true")
        core.read_namespaced_secret.return_value.data["AWS_S3_BUCKET"] = base64.b64encode(b"wrong-bucket").decode()
        with patch("kubernetes.client.CoreV1Api", return_value=core):
            with self.assertRaisesRegex(ValueError, "bucket must equal"):
                wf.deployment_manifests(api)

    def test_missing_runtime_template_lists_installed_autogluon_candidates(self):
        from kubernetes.client.rest import ApiException

        api = Mock()
        api.get_namespaced_custom_object.side_effect = ApiException(status=404)
        api.list_namespaced_custom_object.return_value = {"items": [
            {"metadata": {"name": "installed-autogluon-template"}, "objects": [{
                "kind": "ServingRuntime", "spec": {"supportedModelFormats": [{"name": "autogluon"}]}}]},
            {"metadata": {"name": "other-runtime"}, "objects": []},
        ]}
        with self.assertRaisesRegex(RuntimeError, "AutoGluon template candidates:.*installed-autogluon-template"):
            wf.runtime_template(api)

    def test_missing_runtime_template_handles_forbidden_discovery(self):
        from kubernetes.client.rest import ApiException

        api = Mock()
        api.get_namespaced_custom_object.side_effect = ApiException(status=404)
        api.list_namespaced_custom_object.side_effect = ApiException(status=403)
        with self.assertRaisesRegex(RuntimeError, "listing templates also failed.*403"):
            wf.runtime_template(api)

    def test_runtime_template_forbidden_is_not_reported_as_missing(self):
        from kubernetes.client.rest import ApiException

        api = Mock()
        api.get_namespaced_custom_object.side_effect = ApiException(status=403)
        with self.assertRaises(ApiException):
            wf.runtime_template(api)
        api.list_namespaced_custom_object.assert_not_called()

    def test_deployment_retry_reuses_account_and_creates_remaining_resources(self):
        from kubernetes.client.rest import ApiException

        account = {"apiVersion": "v1", "kind": "ServiceAccount",
                   "metadata": {"name": "fraud-storage", "namespace": "fraud"},
                   "secrets": [{"name": "connection"}], "automountServiceAccountToken": False}
        role = {"apiVersion": "rbac.authorization.k8s.io/v1", "kind": "Role",
                "metadata": {"name": "fraud-storage", "namespace": "fraud"},
                "rules": [{"verbs": ["get"], "resources": ["secrets"], "resourceNames": ["connection"]}]}
        service = {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
                   "metadata": {"name": "fraud-model", "namespace": "fraud"},
                   "spec": {"predictor": {"model": {"runtime": "fraud-runtime"}}}}
        api, core, rbac = Mock(), Mock(), Mock()
        core.create_namespaced_service_account.side_effect = ApiException(status=409)
        core.read_namespaced_service_account.return_value = {
            **account, "metadata": {**account["metadata"], "resourceVersion": "123"},
            "imagePullSecrets": [{"name": "injected-pull-secret"}]}
        with patch("kubernetes.client.CoreV1Api", return_value=core), \
             patch("kubernetes.client.RbacAuthorizationV1Api", return_value=rbac):
            outcomes = wf.create_deployment(api, [account, role, service])
        self.assertEqual([result["action"] for result in outcomes], ["reused", "created", "created"])
        rbac.create_namespaced_role.assert_called_once()
        api.create_namespaced_custom_object.assert_called_once()

    def test_deployment_retry_reuses_matching_custom_resource(self):
        from kubernetes.client.rest import ApiException

        service = {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
                   "metadata": {"name": "fraud-model", "namespace": "fraud",
                                "annotations": {"security.opendatahub.io/enable-auth": "true"}},
                   "spec": {"predictor": {"model": {"runtime": "fraud-runtime"}}}}
        api = Mock()
        api.create_namespaced_custom_object.side_effect = ApiException(status=409)
        api.get_namespaced_custom_object.return_value = {
            **service, "status": {"url": "https://model.example"},
            "spec": {"predictor": {"model": {"runtime": "fraud-runtime", "protocolVersion": "v1"}}}}
        with patch("kubernetes.client.CoreV1Api"), patch("kubernetes.client.RbacAuthorizationV1Api"):
            outcomes = wf.create_deployment(api, [service])
        self.assertEqual(outcomes[0]["action"], "reused")

    def test_deployment_retry_rejects_changed_authentication(self):
        from kubernetes.client.rest import ApiException

        service = {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
                   "metadata": {"name": "fraud-model", "namespace": "fraud",
                                "annotations": {"security.opendatahub.io/enable-auth": "true"}}}
        api = Mock()
        api.create_namespaced_custom_object.side_effect = ApiException(status=409)
        api.get_namespaced_custom_object.return_value = {
            **service, "metadata": {**service["metadata"],
                                    "annotations": {"security.opendatahub.io/enable-auth": "false"}}}
        with patch("kubernetes.client.CoreV1Api"), patch("kubernetes.client.RbacAuthorizationV1Api"):
            with self.assertRaisesRegex(RuntimeError, "Differing fields:.*enable-auth"):
                wf.create_deployment(api, [service])
        api.patch_namespaced_custom_object.assert_not_called()
        api.replace_namespaced_custom_object.assert_not_called()

    def test_deployment_comparison_rejects_other_storage_and_broader_rbac(self):
        self.assertTrue(wf.manifest_differences(
            {"secrets": [{"name": "expected"}]}, {"secrets": [{"name": "other"}]}))
        self.assertTrue(wf.manifest_differences(
            {"rules": [{"verbs": ["get"]}]}, {"rules": [{"verbs": ["get", "list"]}]}))
        self.assertTrue(wf.manifest_differences(
            {"spec": {"storage": {"path": "run-new/predictor"}}},
            {"spec": {"storage": {"path": "run-old/predictor"}}}))

    def test_deployment_forbidden_is_not_reused(self):
        from kubernetes.client.rest import ApiException

        api = Mock()
        api.create_namespaced_custom_object.side_effect = ApiException(status=403)
        service = {"apiVersion": "serving.kserve.io/v1beta1", "kind": "InferenceService",
                   "metadata": {"name": "fraud-model", "namespace": "fraud"}}
        with patch("kubernetes.client.CoreV1Api"), patch("kubernetes.client.RbacAuthorizationV1Api"):
            with self.assertRaises(ApiException):
                wf.create_deployment(api, [service])
        api.get_namespaced_custom_object.assert_not_called()


class NotebookTests(unittest.TestCase):
    def test_notebooks_validate_and_have_valid_python(self):
        root = Path(__file__).resolve().parents[1]
        paths = [*root.glob("*.ipynb"), *(root / "ui").glob("*.ipynb")]
        for path in sorted(paths):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    transformed = "\n".join(line for line in cell.source.splitlines()
                                            if not line.lstrip().startswith("%"))
                    ast.parse(transformed, filename=path.name)


if __name__ == "__main__":
    unittest.main()
