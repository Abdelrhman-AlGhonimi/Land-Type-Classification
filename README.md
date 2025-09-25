# 🌍 Land Type Classification using Sentinel-2/NWPU-RESISC45

A complete deep-learning pipeline to classify land cover types from satellite imagery. The project implements transfer learning baselines (ResNet50) and an improved model (EfficientNetB0), plus a ready-to-run Streamlit demo app.

## 📋 Problem Statement
Accurate land type classification is crucial for agriculture monitoring, urban planning, water resource management, and environmental studies. This work builds and evaluates a robust model for multi-class land cover classification using open-source satellite imagery.

## 🎯 Project Idea & Scope
- Identify categories such as agricultural land, water bodies, urban areas, deserts, roads, and tree cover (45 classes in NWPU-RESISC45).
- Train with transfer learning and fine-tuning; evaluate with accuracy, precision, recall, and F1-score per class.
- Provide a simple web interface (Streamlit) to upload tiles and obtain predictions.

## 🗂️ Dataset
- NWPU-RESISC45 (45 classes, ~31,500 images ≈ 700/class)
- Images are standardized to 224×224 with ImageNet normalization
- Directory expected at `./NWPU-RESISC45/`

Example classes: `airplane, airport, beach, bridge, cloud, desert, forest, ...`

## 🧪 EDA Highlights
- Class distribution is balanced (~700 per class)
- Images are 256×256 (standardized to 224×224 in training)
- Comprehensive color and size analysis included in the notebook

Open the full analysis in the notebook: `GTC.ipynb`

## 🛠️ Models
### 1) Baseline: ResNet50
- ImageNet pre-trained, backbone frozen initially
- Custom classifier head

### 2) Improved: EfficientNetB0
- ImageNet pre-trained backbone
- Custom multi-layer classifier head
- AdamW optimizer, cosine annealing scheduler
- Early stopping + checkpointing to `best_efficientnet_model.pth`

## 📈 Results (from notebook runs)
- ResNet50 Val Accuracy: ~81.9%
- EfficientNetB0 Val Accuracy: ~96.3%

Per-class metrics, confusion matrices, and comparative plots are available in `GTC.ipynb` (sections: Evaluation and Model Comparison).

## 🚀 Quickstart
### 1) Setup
```bash
pip install -r requirements.txt
```

### 2) Train / Evaluate in Jupyter
```bash
jupyter notebook GTC.ipynb
# Run cells sequentially: Data Preparation → EDA → Training → Evaluation
```

### 3) Run the Streamlit Demo
```bash
streamlit run app.py
```
Requirements:
- Place `best_efficientnet_model.pth` in the project root (same folder as `app.py`).
- If `NWPU-RESISC45/` exists locally, the app auto-loads class names; otherwise, it uses a default list.

## 🧩 Project Structure
```text
.
├─ GTC.ipynb                    # Full pipeline: EDA, training, evaluation
├─ app.py                       # Streamlit app (image upload + predictions)
├─ requirements.txt             # Python dependencies
├─ best_efficientnet_model.pth  # Saved fine-tuned weights (place here)
└─ NWPU-RESISC45/               # Dataset (45 classes)
```

## ⚙️ Tech Stack
- Python, PyTorch, torchvision
- EfficientNet-PyTorch
- NumPy, pandas, scikit-learn, matplotlib, seaborn
- Streamlit (demo app)

## 🧪 Repro Tips
- GPU recommended. Adjust batch size if out-of-memory
- Start with LR: backbone 1e-4 / head 1e-3 (AdamW)
- Augmentations: horizontal flip, rotation; consider MixUp/CutMix for further gains

## 🧭 Roadmap
- [ ] Streamlit enhancements (Grad-CAM visualization)
- [ ] Ensembling (ResNet + EfficientNet)
- [ ] EfficientNetV2 / larger variants (B1–B3)
- [ ] Optional EuroSAT/Sentinel-2 support

## 📚 References
- NWPU-RESISC45: http://www.escience.cn/people/gongcheng/NWPU-RESISC45.html
- EfficientNet: https://arxiv.org/abs/1905.11946
- ResNet: https://arxiv.org/abs/1512.03385
- PyTorch: https://pytorch.org/


## 👤 Author

- Abdelrhman Al Ghonimi
- Email : abdelrhmanalghonimi@gmail.com 
- GitHub : https://github.com/Abdelrhman-AlGhonimi
- LinkedIn : https://www.linkedin.com/in/abdelrhman-al-ghonimi-2005902a6/
- portfolio : https://ghonimi.vercel.app/


