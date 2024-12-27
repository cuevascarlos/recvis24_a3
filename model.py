import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, Dinov2ForImageClassification, ResNetForImageClassification
import torch
#nclasses = 500


class Net(nn.Module):
    def __init__(self, nclasses=500):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Dinov2(nn.Module):
    def __init__(self, nclasses=500):
        super(Dinov2, self).__init__()
        #self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = Dinov2ForImageClassification.from_pretrained('facebook/dinov2-base', num_labels=nclasses)

        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x).logits

class Dinov2Giant(nn.Module):
    def __init__(self, nclasses=500):
        super(Dinov2Giant, self).__init__()
        #self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = Dinov2ForImageClassification.from_pretrained('facebook/dinov2-giant', num_labels=nclasses)

        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x).logits

class ResNet(nn.Module):
    def __init__(self, nclasses=500):
        super(ResNet, self).__init__()
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", num_labels=nclasses, ignore_mismatched_sizes=True)
    
        for name, param in self.model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x).logits


class MultiEmbeddingsClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, num_classes=500, dropout_rate=0.2):
        super(MultiEmbeddingsClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim+textual_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, visual_emb, textual_emb):
        x = torch.cat([visual_emb, textual_emb], dim=-1)
        x = self.classifier(x)
        return x


class FusionClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, fusion_dim, num_classes=500, dropout_rate=0.2):
        super(FusionClassifier, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.text_proj = nn.Linear(textual_dim, fusion_dim)
        self.fusion_proj = nn.Linear(2 * fusion_dim, fusion_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(fusion_dim,num_classes)

    def forward(self, visual_emb, text_emb):
        visual_emb_proj = self.visual_proj(visual_emb)
        text_emb_proj = self.text_proj(text_emb)
        combined = torch.cat([visual_emb_proj, text_emb_proj], dim=-1)
        fused = self.fusion_proj(combined)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        fused = self.classifier(fused)
        return fused

class DoubleModel(nn.Module):
    def __init__(self, visual_dim, textual_dim, num_classes=500, dropout_rate=0.2):
        super(DoubleModel, self).__init__()
        self.num_classes = num_classes

        self.visual_classifier = nn.Sequential(
            nn.Linear(visual_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        self.textual_classifier = nn.Sequential(
            nn.Linear(textual_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

        self.weight_layer = nn.Linear(2, 1, bias=False)
        nn.init.constant_(self.weight_layer.weight, 0.5)
    
    def regularization_loss(self):
        # weights >= 0
        positive_penalty = torch.sum(torch.abs(torch.min(self.weight_layer.weight, torch.tensor(0.0, device=self.weight_layer.weight.device))))

        # To sum 1
        weight_sum = self.weight_layer.weight.sum(dim=1)
        normalization_penalty = torch.abs(weight_sum - 1).sum()
        return positive_penalty + normalization_penalty

    def forward(self, visual_emb, textual_emb):

        textual_logits = self.textual_classifier(textual_emb)
        visual_logits  = self.visual_classifier(visual_emb)

        combined_logits = torch.stack([visual_logits, textual_logits], dim=-1)
        output_logits = self.weight_layer(combined_logits).squeeze(-1)
        return output_logits
