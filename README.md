# fraud-detection

URL for Workshop Instructions: <https://rh-aiservices-bu.github.io/fraud-detection/>

An additional [AutoML workflow](automl/README.md) provides linked notebooks for preparing
the same fraud dataset, running the RHOAI tabular managed pipeline, inspecting results,
and deploying an AutoGluon predictor to score Sally's transactions. Start with
[data preparation](automl/00_prepare_data.ipynb); configuration is documented in
[automl/.env.example](automl/.env.example). The existing workshop flow remains available.

For the dashboard walkthrough with screenshots, start with
[AutoML through the UI](automl/ui/00_setup_and_data.ipynb), then deploy through Model Registry
and send the sample prediction request with `curl`.
