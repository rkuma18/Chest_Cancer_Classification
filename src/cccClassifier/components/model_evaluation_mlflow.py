import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
import os
import tempfile
import numpy as np
from cccClassifier.entity.config_entity import EvaluationConfig
from cccClassifier.utils.common import read_yaml, create_directories, save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,
            validation_split=0.30
        )
        
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )
        
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_uri = mlflow.get_tracking_uri()
        tracking_url_type_store = urlparse(tracking_uri).scheme
        
        print(f"MLflow Tracking URI: {tracking_uri}")
        print(f"Tracking backend type: {tracking_url_type_store}")
        
        with mlflow.start_run():
            # Log hyperparameters and evaluation metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1]
            })
            
            # Create input example from validation data
            x, _ = next(self.valid_generator)
            input_example = x[:1]  # Take first image as example
            
            # Create model signature
            input_signature = tf.TensorSpec(
                shape=(None,) + tuple(self.config.params_image_size),
                dtype=tf.float32,
                name="input_1"
            )
            output_signature = tf.TensorSpec(
                shape=(None, self.config.all_params["CLASSES"]),
                dtype=tf.float32,
                name="output_1"
            )
            
            # Handle model logging based on tracking backend
            if "dagshub.com" in tracking_uri:
                # For DagsHub, save model locally and log as artifact
                try:
                    # Create a temporary directory for the model
                    with tempfile.TemporaryDirectory() as temp_dir:
                        model_path = os.path.join(temp_dir, "model.h5")
                        self.model.save(model_path)
                        
                        # Log the model file as an artifact with signature
                        mlflow.keras.log_model(
                            self.model,
                            "model",
                            input_example=input_example,
                            signature=mlflow.models.infer_signature(
                                input_example,
                                self.model.predict(input_example)
                            )
                        )
                        print("Model logged as artifact to DagsHub with signature")
                        
                except Exception as e:
                    print(f"Warning: Could not log model to DagsHub: {str(e)}")
                    print("Continuing without model logging...")
                    
            elif tracking_url_type_store == "file":
                # For local file store, use standard logging with signature
                try:
                    mlflow.keras.log_model(
                        self.model,
                        "model",
                        input_example=input_example,
                        signature=mlflow.models.infer_signature(
                            input_example,
                            self.model.predict(input_example)
                        )
                    )
                    print("Model logged using mlflow.keras.log_model with signature")
                except Exception as e:
                    print(f"Warning: Could not log model: {str(e)}")
                    # Fallback to artifact logging
                    with tempfile.TemporaryDirectory() as temp_dir:
                        model_path = os.path.join(temp_dir, "model.h5")
                        self.model.save(model_path)
                        mlflow.log_artifact(model_path, "model")
                        print("Model logged as artifact (fallback)")
                        
            else:
                # For other MLflow servers, try with model registry
                try:
                    mlflow.keras.log_model(
                        self.model,
                        "model",
                        registered_model_name="VGG16Model",
                        input_example=input_example,
                        signature=mlflow.models.infer_signature(
                            input_example,
                            self.model.predict(input_example)
                        )
                    )
                    print("Model logged with registry and signature")
                except Exception as e:
                    print(f"Warning: Could not log model with registry: {str(e)}")
                    # Fallback to simple logging
                    try:
                        mlflow.keras.log_model(
                            self.model,
                            "model",
                            input_example=input_example,
                            signature=mlflow.models.infer_signature(
                                input_example,
                                self.model.predict(input_example)
                            )
                        )
                        print("Model logged without registry but with signature")
                    except Exception as e2:
                        print(f"Warning: Could not log model at all: {str(e2)}")
                        # Final fallback to artifact logging
                        with tempfile.TemporaryDirectory() as temp_dir:
                            model_path = os.path.join(temp_dir, "model.h5")
                            self.model.save(model_path)
                            mlflow.log_artifact(model_path, "model")
                            print("Model logged as artifact (final fallback)")