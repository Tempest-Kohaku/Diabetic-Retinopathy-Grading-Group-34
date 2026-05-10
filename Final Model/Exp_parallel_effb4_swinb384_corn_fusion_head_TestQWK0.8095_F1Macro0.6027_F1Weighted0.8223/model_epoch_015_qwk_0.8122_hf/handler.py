import base64
import io
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from architecture import ParallelEfficientNetSwin


class EndpointHandler:
    """
    Custom Hugging Face inference handler.

    This class is loaded by Hugging Face when the model is deployed.

    It performs:
        1. model loading
        2. image decoding
        3. preprocessing
        4. model inference
        5. CORN ordinal postprocessing
        6. JSON response formatting

    The output contains:
        - predicted_class
        - label
        - corn_probabilities
        - raw logits
    """

    def __init__(self, path=""):
        self.model_dir = Path(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(self.model_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model = ParallelEfficientNetSwin(
            efficientnet_name=self.config["efficientnet_name"],
            swin_name=self.config["swin_name"],
            num_classes=self.config["num_classes"],
            fusion_hidden_dim=self.config["fusion_hidden_dim"],
            fusion_dropout=self.config["fusion_dropout"],
        )

        state_dict = load_file(str(self.model_dir / "model.safetensors"))
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        image_size = int(self.config["image_size"])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config["normalization_mean"],
                std=self.config["normalization_std"],
            ),
        ])

    def decode_image(self, data):
        """
        Decodes an input image.

        Supported input formats:
            - raw image bytes
            - {"inputs": raw_bytes}
            - {"inputs": base64_string}
        """

        if isinstance(data, bytes):
            image_bytes = data

        elif isinstance(data, dict):
            inputs = data.get("inputs")

            if isinstance(inputs, bytes):
                image_bytes = inputs

            elif isinstance(inputs, str):
                if "," in inputs:
                    inputs = inputs.split(",", 1)[1]

                image_bytes = base64.b64decode(inputs)

            else:
                raise ValueError("Expected inputs to be bytes or a base64 string.")

        else:
            raise ValueError("Expected input data to be bytes or a dictionary.")

        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def corn_prediction_from_logits(logits, threshold=0.5):
        """
        Converts CORN logits into an ordinal class prediction.

        For five classes, the model outputs four logits. Each logit represents
        whether the image passes an ordinal threshold. The number of passed
        thresholds becomes the final predicted class.
        """

        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > threshold).sum(dim=1)
        return prediction, probabilities

    def __call__(self, data):
        image = self.decode_image(data)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(image_tensor)
            prediction, probabilities = self.corn_prediction_from_logits(logits)

        pred_id = int(prediction.item())
        label = self.config["id2label"][str(pred_id)]

        return {
            "predicted_class": pred_id,
            "label": label,
            "corn_probabilities": probabilities.squeeze(0).cpu().tolist(),
            "logits": logits.squeeze(0).cpu().tolist(),
        }
