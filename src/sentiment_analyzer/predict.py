import click

from sentiment_analyzer.model_manager import ModelManager


@click.command()
@click.option('--input_file', '-i', type=click.Path(),
              help='Input file')
@click.option('--output_file', '-o', type=click.Path(),
              help='Output file', required=True)
@click.option('--text', '-t', type=str,
              help='Text to predict')
@click.option('--model_name', '-n', type=str,
              help='Model name', required=True)
@click.option('--model_version', '-v', type=str,
              help='Model version', required=True)
@click.option('--mlflow_url', '-u', type=str, default='http://127.0.0.1:5000',
              help='MLFlow URL', required=True)
def predict(input_file, output_file, text,
            model_name, model_version, mlflow_url):
    model_manager = ModelManager(model_name, model_version, mlflow_url)

    if input_file is not None:
        model_manager.predict_from_file(input_file, output_file)
    elif text is not None:
        model_manager.predict_from_text(text, output_file)
    else:
        raise ValueError('Either input_file or text must be provided')


if '__name__' == '__main__':
    predict()
