import click
import subprocess
import pkg_resources
import mlflow


@click.command()
@click.option('--model_name', '-n', type=str,
              help='Model name', required=True)
@click.option('--model_version', '-v', type=str,
              help='Model version', required=True)
@click.option('--status', '-s', type=str,
              help='Model status', required=True)
def promote(model_name, model_version, status):
    if status not in ['Staging', 'Production', 'Archived']:
        raise ValueError('Status must be one of Staging, Production, Archived')

    if status == 'Production':
        subprocess.run(
            ['pytest',
             pkg_resources.resource_filename(
                 'sentiment_analyzer', '../../tests')]
            )

    client = mlflow.tracking.MlflowClient('http://127.0.0.1:5000')
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=status
    )


if __name__ == '__main__':
    promote()