from models.builders.model_builder import build_model_from_design
from models.segmentation.base_segmentation import BaseSegmentationModel
from core.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("unet_design")
class UNetDesign(BaseSegmentationModel):
    """Design-driven U-Net using modular components."""
    def build_model(self):
        enc_path = self.cfg["model"]["params"]["encoder_design"]
        dec_path = self.cfg["model"]["params"]["decoder_design"]
        self.encoder = build_model_from_design(enc_path)
        self.decoder = build_model_from_design(dec_path)

        inputs = tf.keras.Input(shape=self.cfg["model"]["params"]["input_shape"])
        z = self.encoder(inputs)
        outputs = self.decoder(z)
        self.model = tf.keras.Model(inputs, outputs)
