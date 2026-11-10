# Maintain screenshots in the UI walkthrough

The four notebooks in `automl/ui/` are Markdown-only UI instructions with screenshots. The capture checklist below explains what each image should show. Visible screenshot captions are intentionally omitted from the notebooks; keep capture instructions in this guide.

1. Follow the step on your installed RHOAI release. Capture the screen described in the checklist below.
2. Replace the matching PNG in this folder, keeping its filename. No notebook source edit is needed; reload the notebook to refresh the image.
3. Add a screenshot for each important navigation action, dialog, form, review screen, and verification result. Keep the control and its project/run/model context visible. Use a thin red rectangle around the next click or value to check; number multiple clicks in their actual order. Do not draw over field values or modify observed metrics, statuses, or predictions.
4. If a UI field/menu differs, update the notebook instruction and this checklist to match your captured release. Do not invent controls that are absent.
5. Hide access keys, tokens, Authorization values, private account information, and any credentials before committing the image.
6. Keep actual metrics and outcomes truthful. For failed runs or predictions, capture the observed failure rather than a success example.

Screenshot cells normally display images at a width of 1500 px, shrinking to fit a narrower notebook window while preserving their proportions. The compact terminal response image uses 500 px for readability. Click an image to open the original at full resolution. You can keep high-resolution PNG captures; resizing the files is unnecessary. Images remain separate files so GitHub and Jupyter can render them without embedded attachments. If you rename an image, double-click its Markdown cell and update both the HTML `src` and `href` paths.

Filenames use `<notebook>_<sequence>_<description>.png`, for example `00_1_create_project.png`. The sequence starts at 1 in each notebook and follows the order of its screenshots.

| ID | Filename | Capture |
| --- | --- | --- |
| UI-01 | `00_1_create_project.png` | Show the Create project form or the newly created project, including its display name and namespace where visible. This step precedes Connection and pipeline-server configuration. |
| UI-02 | `00_2_storage_connections.png` | Show the Connections list or non-secret configuration summary, including which bucket each Connection uses. Hide access keys and tokens. |
| UI-02A | `00_3_pipeline_server.png` | Show the project Pipelines tab and highlight Configure pipeline server. The current image shows the entry action; add the configuration form and ready state as follow-up captures below. |
| UI-02B | `00_4_managed_pipeline_definition.png` | Show the installed autogluon-tabular-training-pipeline definition and the installed version shown in its details dialog. This is an inspection step; close the dialog afterward. This does not imply that every managed parameter is exposed by the AutoML wizard. |
| UI-03 | `00_5_training_data_upload.png` | Show the training CSV and its object size in the object-storage console; highlight train.csv. The current image also contains a previous test.csv, which is not selected for this UI route. |
| UI-04 | `01_1_create_automl_experiment.png` | Show the AutoML page, selected project, and action used to create a tabular experiment. Capture the actual button label on this release. |
| UI-05 | `01_2_training_source.png` | Show the selected training Connection/object and fraud target. Include the schema preview if available, with all seven input features. |
| UI-06 | `01_3_search_settings.png` | Show the search settings actually exposed by AutoML, using the labels Run preset, Optimization Metric, and Top models to consider. Record defaults for settings that are not editable; do not invent extra controls. |
| UI-08 | `01_4_experiment_review.png` | Show the native AutoML form with the training input, label, Run preset, Optimization Metric, and Top models to consider. This route uses internal evaluation and has no separate test selection. |
| UI-09 | `01_5_run_progress.png` | Show the submitted run status and task graph. Highlight Run details and the current status. Capture task details/log access where available; do not show tokens or private request headers. |
| UI-10 | `01_6_run_configuration.png` | Show the existing run details and recorded training dataset/configuration. Highlight the run ID and submitted input parameters. This is a read-only check of the submitted run, not another run submission form. |
| UI-11 | `02_1_leaderboard.png` | Show the run ID, candidate names, and the actual ranking metric/score. Highlight the successful run status, model name, and configured metric. Use actual results; do not add a made-up successful score. |
| UI-12 | `02_2_predictor_details.png` | Show the selected model and metrics produced automatically from the training input. Do not imply that a separate evaluation/test dataset was supplied. |
| UI-13 | `02_3_predictor_artifacts.png` | Show the selected model details and the View menu with Save as notebook. Keep the model name and run context visible. Record the complete predictor location from the run artifacts separately if it is not shown in this view. |
| UI-14A | `03_1_create_model_registry.png` | Show the newly created Model Registry and its available status. Hide database passwords and other credentials. |
| UI-14B | `03_2_registry_project_access.png` | Show the registry permissions with the deployment project added. Keep the registry name and project namespace visible; hide credentials. |
| UI-14C | `03_3_register_model.png` | Show the selected model row in the AutoML training results with its three-dot menu and Register model action. If also capturing the registration dialog, show the chosen Model Registry and model/version names. Hide credentials. |
| UI-14 | `03_4_deploy_model.png` | Show AI hub → Models → Registry and the registered model actions menu with Deploy v1 highlighted. The current image shows this navigation action; the deployment form is a separate follow-up capture below. |
| UI-15 | `03_5_model_ready.png` | Show the Ready model, model resource name, and base inference endpoint. Include where access is configured, but never expose a bearer token. |
| UI-16A | `03_6_inference_endpoint.png` | Show the Ready deployment with Internal and external endpoint highlighted. Add the opened endpoint dialog with the external URL copy control as a follow-up capture below. |
| UI-16B | `03_7_model_resource_name.png` | Show the deployed model resource name in the deployment details, sample inference URL, or InferenceService metadata.name view available in your UI. Highlight the exact value used after /v1/models/; hide any credentials. |
| UI-16C | `03_8_inference_token.png` | Show the deployed model access/token controls used to obtain an authorized inference token. Keep the model context visible, but hide the entire token value and any Authorization header. |
| UI-17 | `03_9_prediction_results.png` | Show the terminal output from the curl request: the actual HTTP status and returned predictions for both transactions. Highlight the predictions array and HTTP status while keeping their order visible. Hide tokens and Authorization values; capture a failure honestly if the request failed. |

UI-01–03 (including UI-02A and UI-02B): `00_setup_and_data.ipynb`; UI-04–06 and UI-08–10: `01_create_experiment.ipynb`; UI-11–13: `02_review_results.ipynb`; UI-14A–14C, UI-14–15, UI-16A–16C, and UI-17: `03_deploy_and_predict.ipynb`.

The current AutoML flow accepts only the training dataset (train.csv). Separate evaluation/test data cannot be supplied in this release; that capability is planned for later. The object-storage console and terminal curl request are identified separately from the AutoML screens. UI-16A–16C show where to obtain PREDICT_ENDPOINT, MODEL_NAME, and PREDICT_TOKEN in the UI; always hide the token value. The curl command itself needs no screenshot; UI-17 captures its output in step 7. UI-10 shows the existing run configuration, not an alternative submission route.

## Shared images in the published workshop

The Antora chapter `training-with-automl-ui.adoc` displays these same 23 images beside the relevant procedure and verification steps. `workshop/docs/modules/ROOT/assets/images/automl` is a relative symlink to this directory, so replacing a PNG updates both the notebook and published workshop without maintaining duplicate images. Keep this relative link intact when moving screenshot files.

## Follow-up captures needed

The current images cover the navigation actions and main results. The following dialogs/forms need additional captures from the installed release. Do not replace them with mock screens or link to files before the captures exist. Add each real capture immediately after its corresponding instruction in the notebook and Antora chapter, with a red box around the next control.

| Interaction | Required capture | Place after |
| --- | --- | --- |
| Create input/artifact Connection | S3-compatible Connection form with endpoint, region, and bucket visible; credentials fully hidden; highlight Create | UI-02 |
| Configure pipeline output storage | Pipeline-server form with artifact Connection and bucket visible; highlight the save/configure action, then capture the ready state | UI-02A |
| Inspect the managed definition | Open definition/version details and highlight the close control | UI-02B |
| Register the selected model | Registration dialog with selected registry, model name, version, and final register action | UI-14C |
| Deploy the registered version | Deployment form showing target project, Predictive model, hardware profile, autogluon-1, artifact location, and Require token authentication; highlight Deploy model | UI-14 |
| Copy the external endpoint | Open endpoint dialog with external URL and copy control highlighted | UI-16A |

Use a descriptive suffix for additional filenames, such as `03_4_deploy_model_form.png`, and assign a unique screenshot ID in notebook metadata. Update this checklist and the screenshot count when adding a capture.
