import mlflow
import pandas as pd


class ModelManager:

    def __init__(self, model_name, model_version, url_mlflow):
        '''
        Initialize the model manager
        Params:
            model_name: Model name
            model_version: Model version
            url_mlflow: MLFlow URL
        '''
        mlflow.set_tracking_uri(url_mlflow)
        self.model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

    def predict(self, inputs):
        '''
        Predict sentiment from inputs
        Params:
            inputs: Inputs
        Returns:
            Predicted sentiments
        '''
        return self.model.predict(inputs)

    def predict_from_file(self, input_file, output_file):
        '''
        Predict sentiment from file
        Params:
            input_file: Input file
            output_file: Output file
        '''
        df = pd.read_csv(input_file, index_col=0)
        df['prediction'] = self.predict(df['review'])
        df.to_csv(output_file)

    def predict_from_text(self, text, output_file):
        '''
        Predict sentiment from text
        Params:
            text: Text
            output_file: Output file
        '''
        df = pd.DataFrame([text], columns=['review'])
        df['prediction'] = self.predict(df['review'])
        df.to_csv(output_file)
