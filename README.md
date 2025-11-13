# 🧠 General Vision Framework (GVF)

> **A unified, modular, and extensible deep learning framework** for computer vision — supporting classification, segmentation, self-supervised learning, anomaly detection, and meta-learning — all built on TensorFlow 2.x, fully config-driven, registry-based, and designed for reproducibility and deployment.

---

## 🚀 Overview

This framework was built for **high-performance visual learning** tasks such as:
- **Defect detection** (supervised, self-supervised, and anomaly-based)
- **Semantic segmentation**
- **Representation learning** (SimCLR, SupCon)
- **Meta-learning and few-shot learning** (ProtoNet, MAML)
- **Lightweight deployment and compression**

It emphasizes:
- 🧩 **Modularity** – every component is plug-and-play  
- ⚙️ **Config-driven orchestration** – no code changes required  
- 🧠 **Stable training and reproducibility**  
- 📊 **Continuous diagnostics and auto-reporting**  
- 🚀 **Production-ready export and inference**

---

## 📂 Project Structure

```
cv_framework/
├── core/
│   ├── base_model.py              # Abstract base for all models
│   ├── base_dataset.py            # Unified dataset interface
│   ├── builder.py                 # Builds from YAML configs
│   ├── registry.py                # Global registries
│   ├── utils/
│   │   ├── seed_utils.py
│   │   ├── memory_utils.py        # Memory management
│   │   ├── logger.py
│   │   └── metrics.py
│
├── models/
│   ├── backbones/
│   │   ├── lightcnn_tf.py
│   │   ├── model_components.py    # Modular CNN, Residual, Attention blocks
│   ├── segmentation/
│   │   ├── base_segmentation.py
│   │   ├── unet_tf.py
│   │   └── (DeepLab, UNet++ etc.)
│   ├── self_supervised/
│   │   ├── simclr_tf.py
│   │   ├── supcon_tf.py
│   ├── anomaly/
│   │   ├── autoencoder_tf.py
│   │   ├── ganomaly_tf.py
│   │   └── fanogan_tf.py
│   ├── meta_learning/
│   │   ├── base_meta.py
│   │   ├── protonet_tf.py
│   │   └── maml_tf.py
│   └── builders/
│       └── model_builder.py       # YAML/JSON design builder
│
├── data/
│   ├── datasets/
│   │   ├── defect_dataset_tf.py
│   ├── meta_tasks/
│   │   └── episodic_loader_tf.py
│
├── training/
│   ├── base_trainer.py
│   ├── hypersearch/
│   │   ├── optuna_search.py
│   │   ├── grid_search.py
│   │   └── random_search.py
│
├── evaluation/
│   ├── sanity_checks/
│   │   ├── generic.py
│   │   ├── ssl_checks.py
│   │   ├── anomaly_checks.py
│   │   ├── meta_checks.py
│   │   ├── diagnostics_manager.py
│   │   ├── visualization_utils.py
│   │   ├── logger.py
│   │   └── report_generator.py
│
├── experiments/
│   ├── manager/
│   │   ├── experiment_manager.py
│   │   ├── environment_utils.py
│   │   ├── tracker.py
│   │   └── run_summary.py
│
├── deployment/
│   ├── exporter_tf.py
│   ├── compression_utils.py
│   ├── version_manager.py
│   ├── inference_api.py
│   └── compression_report.py
│
├── configs/
│   ├── train_lightcnn.yaml
│   ├── train_unet.yaml
│   ├── train_supcon.yaml
│   ├── train_autoencoder.yaml
│   ├── train_protonet.yaml
│   ├── models/
│   │   ├── light_cnn_design.yaml
│   │   ├── unet_encoder.yaml
│   │   ├── unet_decoder.yaml
│   │   ├── discriminator_basic.yaml
│   │   └── projection_head.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── export_model.py
│   └── run_hypersearch.py
│
└── README.md
```

---

## 🧩 Core Features

### 🔧 **Registry + Config-Driven Architecture**
- Every model, dataset, loss, optimizer, and scheduler is registered via decorators.
- All experiments are defined in YAML (no code edits).

```python
@MODEL_REGISTRY.register("lightcnn")
class LightCNN(BaseModel): ...
```

---

### 🧱 **Reusable Building Blocks (`model_components.py`)**
Includes:
- `ConvBlock`, `ResidualBlock`, `DenseBlock`
- `SEBlock`, `CBAM`, `SelfAttention2D`, `SpatialAttentionLite`
- Fully compatible with YAML-based designs.

> **Performance Gains:** Attention modules improve representation focus and convergence stability (Hu et al., 2018; Woo et al., 2018; Wang et al., 2018).

---

### 🧠 **Supported Model Families**
| Category | Examples | Notes |
|-----------|-----------|-------|
| Classification | LightCNN, MobileNet | Lightweight backbones |
| Segmentation | U-Net, YAML-based encoder/decoder | Supports Dice, Focal, Tversky |
| SSL | SimCLR, SupCon, BYOL | Modular projection heads |
| Anomaly | Autoencoder, GANomaly, f-AnoGAN | Design-driven encoders/decoders |
| Meta-learning | ProtoNet, MAML, Reptile | Uses episodic loaders |
| Attention | SEBlock, CBAM, Self-Attention | Drop-in YAML layers |

---

### 🔍 **Hyperparameter & Architecture Search**
- Integrated **Optuna**, **Grid**, and **Random** search.
- Architecture-level NAS via YAML design swapping.
- Automatic logging of best trials under `experiments/hypersearch/`.

---

### 📊 **Diagnostics & Continuous Evaluation**
- Built-in sanity checks for every model type:
  - Gradient health, forward pass, overfit-one-batch
  - SSL: Inter/intra-class embedding distances
  - Anomaly: Reconstruction loss visualization
  - Meta: Prototype embedding plots
- Continuous diagnostics every *N* epochs (TensorBoard + PDF reports).

---

### 💾 **Experiment Management & Tracking**
- Environment snapshot (`environment.json`)
- Config archive (`config_snapshot.json`)
- Training logs (TensorBoard + JSON)
- Auto-generated run summaries
- Parallel multi-config orchestration

---

### ⚡ **Memory & Resource Management**
- `clear_tf_memory()` prevents GPU memory leaks.
- Dataset cache clearing for sequential runs.
- Optional GPU monitor for diagnostics.

---

### 🧬 **Compression & Deployment**
- Pruning, Quantization (INT8), Clustering.
- TF SavedModel, ONNX, and TFLite export.
- Unified inference API (`inference_api.py`):
  ```python
  api = InferenceAPI(config_path, weights_path)
  preds = api.classify_image("sample.jpg")
  ```

> Compression typically yields **4–10× smaller models**, **1.5–3× faster inference**, ≤1–2% accuracy change (Han et al., 2015; Jacob et al., 2018).

---

### 🧠 **Reproducibility**
- Random seed control  
- Full environment capture  
- Config + code version snapshots  
- Model registry versioning (`model_registry.json`)

---

## 🧩 Typical Workflow

1. **Design the architecture**
   ```yaml
   configs/models/my_custom_design.yaml
   ```
2. **Define experiment config**
   ```yaml
   configs/train_custom.yaml
   ```
3. **Run training**
   ```bash
   python scripts/train.py --config configs/train_custom.yaml
   ```
4. **Run diagnostics**
   ```bash
   tensorboard --logdir experiments/runs/
   ```
5. **Compress & export**
   ```bash
   python scripts/export_model.py --config experiments/runs/<run>/config_snapshot.json
   ```
6. **Deploy for inference**
   ```python
   from deployment.inference_api import InferenceAPI
   api = InferenceAPI("config.yaml", "final_model/variables/variables")
   api.predict(input_tensor)
   ```

---

## 📚 References

- Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks.* CVPR.  
- Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). *CBAM: Convolutional Block Attention Module.* ECCV.  
- Wang, X. et al. (2018). *Non-local Neural Networks.* CVPR.  
- Han, S. et al. (2015). *Deep Compression.* NIPS.  
- Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR.  

---

## ✅ Summary

**You now have a unified TensorFlow 2.x framework** for:
- 🧱 Modular deep learning experiments  
- 🧩 Extensible vision architectures  
- 📊 Continuous diagnostics and automatic reporting  
- ⚙️ Automated search and orchestration  
- 🚀 Deployment-ready compression and export  

This foundation can be extended indefinitely — from attention-based segmentation to contrastive meta-learning — all through configuration files, not rewrites.
