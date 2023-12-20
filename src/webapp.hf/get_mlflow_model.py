import click
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient


@click.command()
@click.option("--mlflow_server_uri")
@click.option("--model_name")
@click.option("--model_version")
@click.option("--target_path")
def main(mlflow_server_uri, model_name, model_version, target_path):
    '''
    Get model from MLFlow server
    Params:
        mlflow_server_uri: MLFlow server URI
        model_name: Model name
        model_version: Model version
        target_path: Target path
    '''
    mlflow.set_tracking_uri(mlflow_server_uri)
    model = mlflow.sklearn.load_model(
        model_uri=f'models:/{model_name}/{model_version}')
    mlflow.sklearn.save_model(model, target_path)

    client = MlflowClient()

    model_details = client.get_model_version(
        name=model_name,
        version=model_version
    )

    with open(f'{target_path}/model_infos.json', 'w') as f:
        f.write(str(model_details.__dict__))


if __name__ == "__main__":
    main()
