import mlflow
import pandas as pd


class ModelManager:

    def __init__(self, model_name, model_version, url_mlflow):
        mlflow.set_tracking_uri(url_mlflow)
        self.model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_version}"
        )

    def predict(self, inputs):
        return self.model.predict(inputs)

    def predict_from_file(self, input_file, output_file):
        df = pd.read_csv(input_file, index_col=0)
        df['prediction'] = self.predict(df['review'])
        df.to_csv(output_file)

    def predict_from_text(self, text, output_file):
        df = pd.DataFrame([text], columns=['review'])
        df['prediction'] = self.predict(df['review'])
        df.to_csv(output_file)
