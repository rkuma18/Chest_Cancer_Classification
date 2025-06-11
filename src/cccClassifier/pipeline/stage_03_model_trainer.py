from cccClassifier.config.configuration import ConfigurationManager
from cccClassifier.components.model_trainer import Training
from cccClassifier import logger
import tensorflow as tf
import json
import os

STAGE_NAME = "Training"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(os.path.dirname(training_config.trained_model_path), 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]

        # Train the model
        training.train(callback_list=callbacks)

        # Evaluate the model
        metrics = training.evaluate_model()
        
        # Save metrics
        metrics_path = os.path.join(os.path.dirname(training_config.trained_model_path), 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 