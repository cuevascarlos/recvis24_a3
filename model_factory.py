"""Python file to instantite the model and the transform that goes with it."""

from data import data_transforms, data_transforms_transformers
from model import Net, Dinov2, Dinov2Giant, ResNet, MultiEmbeddingsClassifier, FusionClassifier, DoubleModel
from transformers import AutoImageProcessor


class ModelFactory:
    def __init__(self, model_name: str, nclasses: int = 500):
        self.model_name = model_name
        self.nclasses = nclasses
        self.model = self.init_model()
        self.transform = self.init_transform()

    def init_model(self):
        if self.model_name == "basic_cnn":
            return Net(nclasses=self.nclasses)
        elif self.model_name == "dino":
            return Dinov2(nclasses=self.nclasses)
        elif self.model_name == "dino_giant":
            return Dinov2Giant(nclasses=self.nclasses)
        elif self.model_name == "ResNet":
            return ResNet(nclasses=self.nclasses)
        elif self.model_name == "multi-embeddings":
            return MultiEmbeddingsClassifier(visual_dim=768, textual_dim=768, num_classes=self.nclasses)
        elif self.model_name == "fusion":
            return FusionClassifier(visual_dim=768, textual_dim=768,fusion_dim=512, num_classes=self.nclasses)
        elif self.model_name == "double":
            return DoubleModel(visual_dim=768, textual_dim=768, num_classes=self.nclasses)
        else:
            raise NotImplementedError("Model not implemented")

    def init_transform(self):
        if self.model_name == "basic_cnn":
            return data_transforms
        elif self.model_name == "dino":
            return data_transforms_transformers
        elif self.model_name == "dino_giant":
            return data_transforms_transformers
        elif self.model_name == "ResNet":
            return data_transforms_transformers
        elif self.model_name == "multi-embeddings":
            pass
        elif self.model_name == "fusion":
            pass
        elif self.model_name == "double":
            pass
        else:
            raise NotImplementedError("Transform not implemented")

    def get_model(self):
        return self.model

    def get_transform(self):
        return self.transform

    def get_all(self):
        return self.model, self.transform
