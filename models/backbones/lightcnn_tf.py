from models.builders.model_builder import build_model_from_design
from core.base_model import BaseModel
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("lightcnn_design")
class LightCNNDesign(BaseModel):
    """LightCNN built from YAML model design."""
    def build_model(self):
        design_path = self.cfg["model"]["params"]["design_path"]
        self.model = build_model_from_design(design_path)

    def forward(self, inputs, training=False):
        return self.model(inputs, training=training)
