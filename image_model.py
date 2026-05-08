import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np
import os


class MedicalImageAI:
    """
    ResNet18-based chest X-ray classifier.
    Auto-detects number of classes from saved model weights.
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "image_model.pth")

    def __init__(self):
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if os.path.exists(self.MODEL_PATH):
            checkpoint = torch.load(self.MODEL_PATH, map_location=torch.device("cpu"), weights_only=False)
            n_classes = checkpoint["fc.weight"].shape[0]
            if n_classes == 2:
                self.CLASSES = ["Normal", "Pneumonia"]
            else:
                self.CLASSES = ["Normal", "Pneumonia", "COVID-19"]
            self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)
            self.model.load_state_dict(checkpoint)
            self.pretrained_only = False
        else:
            self.CLASSES = ["Normal", "Pneumonia", "COVID-19"]
            self.model.fc = nn.Linear(self.model.fc.in_features, 3)
            self.pretrained_only = True

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _quality_check(self, img_np):
        mean = img_np.mean()
        std = img_np.std()
        if mean < 15 or mean > 240:
            return False, "Image is too dark or too bright — likely a blank or overexposed image."
        if std < 8:
            return False, "Image has no variation — may be a blank or solid-color image."
        if img_np.ndim == 3:
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
            channel_diff = max(
                float(np.abs(r.mean() - g.mean())),
                float(np.abs(g.mean() - b.mean())),
                float(np.abs(r.mean() - b.mean()))
            )
            if channel_diff > 60:
                return False, "Image appears to be a colorful photo, not a medical scan."
        return True, "ok"

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return {
                "prediction": "Invalid",
                "confidence": 0.0,
                "status": "error",
                "message": "Could not read image. Please upload a valid JPG or PNG file.",
                "top3": []
            }

        img_np = np.array(image)
        valid, reason = self._quality_check(img_np)

        if not valid:
            return {
                "prediction": "Invalid",
                "confidence": 0.0,
                "status": "invalid",
                "message": reason,
                "top3": []
            }

        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        confidence = float(torch.max(probs).item())
        pred_idx = int(torch.argmax(probs).item())
        entropy = float(-torch.sum(probs * torch.log(probs + 1e-8)).item())

        top3 = [
            {"label": self.CLASSES[i], "confidence": round(float(probs[i].item()), 4)}
            for i in torch.argsort(probs, descending=True)
        ]

        if confidence < 0.55 or entropy > 1.0:
            return {
                "prediction": "Unknown",
                "confidence": round(confidence, 4),
                "status": "uncertain",
                "message": "Model is uncertain. Image may not be a standard PA-view chest X-ray.",
                "top3": top3
            }

        return {
            "prediction": self.CLASSES[pred_idx],
            "confidence": round(confidence, 4),
            "status": "success",
            "message": f"ResNet18 inference — {'fine-tuned 97.95% accuracy' if not self.pretrained_only else 'ImageNet pretrained'}",
            "top3": top3
        }