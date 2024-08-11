from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging import logger
import os


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()