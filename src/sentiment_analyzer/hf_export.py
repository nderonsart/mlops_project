import click
import pandas as pd
import mlflow
from skops import hub_utils
import tempfile
import sklearn


@click.command()
@click.option('--model_name', '-n', type=str,
              help='Model name', required=True)
@click.option('--model_version', '-v', type=str,
              help='Model version', required=True)
@click.option('--mlflow_url', '-u', type=str, default='http://127.0.0.1:5000',
              help='MLFlow URL', required=True)
@click.option('--data_file', '-d', type=str,
              help='Data file', required=True)
@click.option('--hf_id', '-i', type=str,
              help='HuggingFace ID', required=True)
@click.option('--hf_token', '-t', type=str,
              help='HuggingFace token', required=True)
def hf_export(model_name, model_version, mlflow_url,
              data_file, hf_id, hf_token):
    '''
    Export model to HuggingFace Hub
    Params:
        model_name: Model name
        model_version: Model version
        mlflow_url: MLFlow URL
        data_file: Data file
        hf_id: HuggingFace ID
        hf_token: HuggingFace token
    '''
    mlflow.set_tracking_uri(mlflow_url)
    model = mlflow.sklearn.load_model(
        model_uri=f'models:/{model_name}/{model_version}')

    data = pd.read_csv(data_file)

    with tempfile.TemporaryDirectory() as tmp_path:
        mlflow.sklearn.save_model(model, tmp_path)

        f = open(tmp_path + "/requirements.txt", "r")
        requirements = f.readlines()

        requirements.append(f"scikit-learn={sklearn.__version__}")
        hub_utils.init(
            model=tmp_path+"/model.pkl",
            requirements=requirements,
            dst=tmp_path + '/tmp',
            task="text-classification",
            data=data,
        )
        hub_utils.push(
            repo_id=hf_id+"/the-very-best-model",
            token=hf_token,
            source=tmp_path+'/tmp',
        )


if __name__ == '__main__':
    hf_export()
