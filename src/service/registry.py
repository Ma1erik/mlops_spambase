import os
import json
import hashlib
import threading
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import torch
import torch.nn as nn


def calculate_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


class ModelWrapper:
    def __init__(
        self,
        model_type: str,
        model_obj: Any,
        scaler_mean: Any = None,
        scaler_scale: Any = None,
    ):
        self.model_type = model_type
        self.model_obj = model_obj
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale

    def predict(self, features: list[float]) -> Tuple[int, float]:
        X = np.array([features], dtype=np.float32)
        if self.model_type == "ml_joblib":
            pred = int(self.model_obj.predict(X)[0])
            if hasattr(self.model_obj, "predict_proba"):
                prob = float(self.model_obj.predict_proba(X)[0][1])
            else:
                prob = float(pred)
            return pred, prob
        elif self.model_type == "dl_pt":
            if self.scaler_mean is not None and self.scaler_scale is not None:
                denom = np.where(self.scaler_scale == 0, 1.0, self.scaler_scale)
                X = (X - self.scaler_mean) / denom
            self.model_obj.eval()
            with torch.no_grad():
                tensor_x = torch.from_numpy(X.astype(np.float32))
                logits = self.model_obj(tensor_x)
                if logits.shape[-1] == 1 or len(logits.shape) == 1:
                    prob = float(torch.sigmoid(logits.squeeze()).item())
                else:
                    prob = float(torch.softmax(logits, dim=1)[0][1].item())
                pred = 1 if prob >= 0.5 else 0
            return pred, prob
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


class ModelRegistry:
    def __init__(self, registry_file: str):
        self.registry_file = registry_file
        self._lock = threading.Lock()
        self._loaded_models: Dict[str, ModelWrapper] = {}
        self.active_version: Optional[str] = None
        self.previous_version: Optional[str] = None
        self.registry_data: Dict[str, Any] = {}
        self.load_registry()

    def load_registry(self) -> None:
        with open(self.registry_file, "r", encoding="utf-8") as f:
            self.registry_data = json.load(f)

        self.active_version = self.registry_data.get("active_version")
        self.previous_version = self.registry_data.get("previous_version")

        for ver, info in self.registry_data.get("models", {}).items():
            if info["status"] in ["рабочая", "пробная", "готовится"]:
                self._load_and_verify_model(ver, info)

    def _load_and_verify_model(self, version: str, info: Dict[str, Any]) -> None:
        path = info["path"]
        expected_sha = info["sha256"]
        actual_sha = calculate_sha256(path)

        if actual_sha != expected_sha:
            raise ValueError(
                f"SHA256 mismatch for model {version}: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        model_type = info["model_type"]
        scaler_mean = None
        scaler_scale = None

        if model_type == "ml_joblib":
            model_obj = joblib.load(path)
        elif model_type == "dl_pt":
            checkpoint = torch.load(
                path, map_location=torch.device("cpu"), weights_only=False
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                if (
                    "scaler_mean" in checkpoint
                    and checkpoint["scaler_mean"] is not None
                ):
                    scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
                if (
                    "scaler_scale" in checkpoint
                    and checkpoint["scaler_scale"] is not None
                ):
                    scaler_scale = np.array(
                        checkpoint["scaler_scale"], dtype=np.float32
                    )

                try:
                    from src.train_dl import SpamMLP

                    try:
                        model_obj = SpamMLP()
                    except TypeError:
                        try:
                            model_obj = SpamMLP(checkpoint.get("config"))
                        except Exception:
                            model_obj = SpamMLP(57)
                    model_obj.load_state_dict(state_dict)
                except Exception:
                    out_dim = state_dict["net.6.weight"].shape[0]

                    class FallbackSpamMLP(nn.Module):
                        def __init__(self, out_features):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(57, 64),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(32, out_features),
                            )

                        def forward(self, x):
                            return self.net(x)

                    model_obj = FallbackSpamMLP(out_dim)
                    model_obj.load_state_dict(state_dict)
            else:
                model_obj = checkpoint
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self._loaded_models[version] = ModelWrapper(
            model_type, model_obj, scaler_mean, scaler_scale
        )

    def get_active_model(self) -> Tuple[str, ModelWrapper]:
        with self._lock:
            if (
                not self.active_version
                or self.active_version not in self._loaded_models
            ):
                raise RuntimeError("No active model loaded in registry")
            return self.active_version, self._loaded_models[self.active_version]

    def promote(self, target_version: str, reason: str) -> None:
        with self._lock:
            if target_version not in self.registry_data["models"]:
                raise KeyError(f"Version {target_version} not found in registry")

            if target_version not in self._loaded_models:
                self._load_and_verify_model(
                    target_version, self.registry_data["models"][target_version]
                )

            curr_active = self.active_version
            if curr_active == target_version:
                return

            if curr_active:
                self.registry_data["models"][curr_active]["status"] = "пробная"
                self.previous_version = curr_active

            self.registry_data["models"][target_version]["status"] = "рабочая"
            self.registry_data["models"][target_version][
                "last_transition_reason"
            ] = reason
            self.active_version = target_version
            self.registry_data["active_version"] = self.active_version
            self.registry_data["previous_version"] = self.previous_version

            self._save_registry_file()

    def rollback(self, reason: str) -> str:
        with self._lock:
            if (
                not self.previous_version
                or self.previous_version not in self.registry_data["models"]
            ):
                raise RuntimeError("No previous version recorded for rollback")

            prev = self.previous_version
            curr = self.active_version

            self.registry_data["models"][curr]["status"] = "снята"
            self.registry_data["models"][prev]["status"] = "рабочая"
            self.registry_data["models"][prev][
                "last_transition_reason"
            ] = f"Rollback: {reason}"

            self.active_version = prev
            self.previous_version = curr
            self.registry_data["active_version"] = self.active_version
            self.registry_data["previous_version"] = self.previous_version

            self._save_registry_file()
            return self.active_version

    def _save_registry_file(self) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(self.registry_data, f, ensure_ascii=False, indent=2)
