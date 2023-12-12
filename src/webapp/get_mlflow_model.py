import click
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

'''
Model details

Vous pouvez modifier votre script pour qu'il charge également les informations associées au modèles, les model details. 
Pour cela vous pouvez utiliser mlflow.get_model_version, et stocker le résultat en json dans le conteneur au même titre que le modèle.

Vous pouvez alors :
Ajouter un endpoint get_details qui renvoie les détail du modèle utilisé par le conteneur.
Ajouter un endpoint get_stage qui renvoie le stage level, à savoir None, Staging, Production ou Archive.
'''


@click.command()
@click.option("--mlflow_server_uri")
@click.option("--model_name")
@click.option("--model_version")
@click.option("--target_path")
def main(mlflow_server_uri, model_name, model_version, target_path):
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
