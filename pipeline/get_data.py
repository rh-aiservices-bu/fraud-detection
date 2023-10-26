from kfp.components import OutputPath


def get_data(output_path: OutputPath()):
    import urllib.request
    print("starting download...")
    url = "https://github.com/cfchase/fraud-detection-notebooks/raw/main/data/card_transdata.csv"
    urllib.request.urlretrieve(url, output_path)
    print("done")

