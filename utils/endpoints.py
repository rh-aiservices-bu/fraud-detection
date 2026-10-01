import requests
import warnings
from urllib.parse import urlparse


def find_port(base_url, model_name, ports=[8888, 8080, None]):
    """Try common ports and return the first working inference endpoint base URL.

    If base_url already includes a port, that URL is tried directly without
    iterating through the default port list.
    """
    test_payload = {
        "inputs": [{
            "name": "dense_input",
            "shape": [1, 5],
            "datatype": "FP32",
            "data": [0.0, 1.0, 1.0, 1.0, 0.0]
        }]
    }
    if urlparse(base_url).port is not None:
        ports = [None]  # port already in URL, try as-is

    for port in ports:
        url = f"{base_url}:{port}" if port else base_url
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resp = requests.post(
                    f"{url}/v2/models/{model_name}/infer",
                    json=test_payload,
                    verify=False,
                    timeout=5
                )
            if resp.status_code == 200:
                print(f"Working endpoint found: {url}")
                return url
            else:
                print(f"  Port {port or 'none'}: HTTP {resp.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"  Port {port or 'none'}: connection refused")
        except requests.exceptions.Timeout:
            print(f"  Port {port or 'none'}: timed out")
    raise RuntimeError("No working port found. Check your endpoint URL and that the model is deployed.")
