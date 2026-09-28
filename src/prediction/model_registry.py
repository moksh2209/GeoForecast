import json
import joblib
import os

class ModelRegistry:
    def __init__(self, registry_path='models/production'):
        self.registry_path = registry_path
        self.models = {}
        self.metadata = {}
        self._load_registry()

    def _load_registry(self):
        levels = ['maharashtra', 'district']
        for level in levels:
            meta_path = os.path.join(self.registry_path, level, 'metadata.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    self.metadata[level] = json.load(f)
            
            # We don't load models into memory yet to save space, only load on demand
            self.models[level] = {}

    def get_model(self, level, horizon):
        h_key = f'h{horizon}'
        if h_key not in self.models[level]:
            model_path = os.path.join(self.registry_path, level, f'model_{h_key}.joblib')
            if not os.path.exists(model_path):
                raise ValueError(f"Model for {level} horizon {horizon} not found at {model_path}")
            self.models[level][h_key] = joblib.load(model_path)
            
        return self.models[level][h_key]

    def get_metadata(self, level, horizon):
        h_key = f'h{horizon}'
        return self.metadata.get(level, {}).get(h_key, {})
