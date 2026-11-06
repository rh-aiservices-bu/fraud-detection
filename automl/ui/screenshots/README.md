# Maintain screenshots in the UI walkthrough

The four notebooks in `automl/ui/` are Markdown-only UI instructions with screenshots. The capture checklist below explains what each image should show. Visible screenshot captions are intentionally omitted from the notebooks; keep capture instructions in this guide.

1. Follow the step on your installed RHOAI release. Capture the screen described in the checklist below.
2. Replace the matching PNG in this folder, keeping its filename. No notebook source edit is needed; reload the notebook to refresh the image.
3. If a UI field/menu differs, update the notebook instruction and this checklist to match your captured release. Do not invent controls that are absent.
4. Hide access keys, tokens, Authorization values, private account information, and any credentials before committing the image.
5. Keep actual metrics and outcomes truthful. For failed runs or predictions, capture the observed failure rather than a success example.

Screenshot cells normally display images at a width of 1500 px, shrinking to fit a narrower notebook window while preserving their proportions. The compact terminal response image uses 500 px for readability. Click an image to open the original at full resolution. You can keep high-resolution PNG captures; resizing the files is unnecessary. Images remain separate files so GitHub and Jupyter can render them without embedded attachments. If you rename an image, double-click its Markdown cell and update both the HTML `src` and `href` paths.

Filenames use `<notebook>_<sequence>_<description>.png`, for example `00_1_create_project.png`. The sequence starts at 1 in each notebook and follows the order of its screenshots.

| ID | Filename | Capture |
| --- | --- | --- |
| UI-01 | `00_1_create_project.png` | Show the Create project form or the newly created project, including its display name and namespace where visible. This step precedes Connection and pipeline-server configuration. |
| UI-02 | `00_2_storage_connections.png` | Show the Connections list or non-secret configuration summary, including which bucket each Connection uses. Hide access keys and tokens. |
| UI-02A | `00_3_pipeline_server.png` | Show pipeline-server configuration with the artifact Connection selected and the output bucket visible. Hide storage credentials; capture the ready state as a second image if helpful. |
| UI-02B | `00_4_managed_pipeline_definition.png` | Show the installed autogluon-tabular-training-pipeline definition and its selected version. This does not imply that every managed parameter is exposed by the AutoML wizard. |
| UI-03 | `00_5_training_data_upload.png` | Show the completed training CSV upload and its object size. Only training input is selected by the current AutoML wizard; this is the object-storage console. |
| UI-04 | `01_1_create_automl_experiment.png` | Show the AutoML page, selected project, and action used to create a tabular experiment. Capture the actual button label on this release. |
| UI-05 | `01_2_training_source.png` | Show the selected training Connection/object and fraud target. Include the schema preview if available, with all seven input features. |
| UI-06 | `01_3_search_settings.png` | Show the search settings actually exposed by AutoML, such as the preset and ranking metric. Record defaults for settings that are not editable; do not invent extra controls. |
| UI-08 | `01_4_experiment_review.png` | Show the native AutoML form with the training input, label, preset, and actual metric settings. This route uses internal evaluation and has no separate test selection. |
| UI-09 | `01_5_run_progress.png` | Show the submitted run ID and status. Capture task details/log access where available; do not show tokens or private request headers. |
| UI-10 | `01_6_run_configuration.png` | Show the existing run details and recorded training dataset/configuration. This is a read-only check of the submitted run, not another run submission form. |
| UI-11 | `02_1_leaderboard.png` | Show the run ID, candidate names, and the actual ranking metric/score. Use actual results; do not add a made-up successful score. |
| UI-12 | `02_2_predictor_details.png` | Show the selected model and metrics produced automatically from the training input. Do not imply that a separate evaluation/test dataset was supplied. |
| UI-13 | `02_3_predictor_artifacts.png` | Show the selected model details and the View menu with Save as notebook. Keep the model name and run context visible. Record the complete predictor location from the run artifacts separately if it is not shown in this view. |
| UI-14A | `03_1_create_model_registry.png` | Show the newly created Model Registry and its available status. Hide database passwords and other credentials. |
| UI-14B | `03_2_registry_project_access.png` | Show the registry permissions with the deployment project added. Keep the registry name and project namespace visible; hide credentials. |
| UI-14C | `03_3_register_model.png` | Show the selected model row in the AutoML training results with its three-dot menu and Register model action. If also capturing the registration dialog, show the chosen Model Registry and model/version names. Hide credentials. |
| UI-14 | `03_4_deploy_model.png` | Show the Deploy v1 form with Model type set to Predictive model, the selected Hardware profile (for example default-profile), Model framework autogluon-1, and Model access with Require token authentication selected. Include the target project and registered artifact location where visible; hide credentials. |
| UI-15 | `03_5_model_ready.png` | Show the Ready model, model resource name, and base inference endpoint. Include where access is configured, but never expose a bearer token. |
| UI-16A | `03_6_inference_endpoint.png` | Show the Ready deployment and its external Inference endpoint with the copy control. Keep enough project/model context to identify the deployment. |
| UI-16B | `03_7_model_resource_name.png` | Show the deployed model resource name in the deployment details, sample inference URL, or InferenceService metadata.name view available in your UI. Highlight the exact value used after /v1/models/; hide any credentials. |
| UI-16C | `03_8_inference_token.png` | Show the deployed model access/token controls used to obtain an authorized inference token. Keep the model context visible, but hide the entire token value and any Authorization header. |
| UI-17 | `03_9_prediction_results.png` | Show the terminal output from the curl request: the actual HTTP status and returned predictions for both transactions. Keep their order visible. Hide tokens and Authorization values; capture a failure honestly if the request failed. |

UI-01–03 (including UI-02A and UI-02B): `00_setup_and_data.ipynb`; UI-04–06 and UI-08–10: `01_create_experiment.ipynb`; UI-11–13: `02_review_results.ipynb`; UI-14A–14C, UI-14–15, UI-16A–16C, and UI-17: `03_deploy_and_predict.ipynb`.

The current AutoML flow accepts only the training dataset (train.csv). Separate evaluation/test data cannot be supplied in this release; that capability is planned for later. The object-storage console and terminal curl request are identified separately from the AutoML screens. UI-16A–16C show where to obtain PREDICT_ENDPOINT, MODEL_NAME, and PREDICT_TOKEN in the UI; always hide the token value. The curl command itself needs no screenshot; UI-17 captures its output in step 7. UI-10 shows the existing run configuration, not an alternative submission route.
