<div align="center">

# Modular CNN

### Handwritten character recognition with PyTorch Lightning

A compact, hands-on CNN project for learning the end-to-end workflow behind image classification—from EMNIST data loading and model training to TorchScript export.

[Explore the notebook](updated_CNN.ipynb) · [Browse the model files](MODEL/)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Lightning-EE4C2C?logo=pytorch&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-EMNIST-6C5CE7)
![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)

</div>

---

## What this project does

This repository builds and trains a modular convolutional neural network to classify handwritten characters from the **EMNIST ByClass** dataset. The 62 output classes cover digits, uppercase letters, and lowercase letters.

The project is intentionally educational: the notebook walks through the model, training and validation steps, experiment logging, checkpointing, and exporting a trained model as TorchScript. It is a learning project—not a claim of state-of-the-art accuracy.

## Highlights

- **Modular CNN:** two convolutional blocks with batch normalization, ReLU activations, and max pooling.
- **62-class classification:** trained on grayscale, 28 × 28 EMNIST images.
- **Lightning training loop:** validation metrics, early stopping, and best-checkpoint saving.
- **Experiment tracking:** optional Weights & Biases logging.
- **TorchScript artifacts:** exported model files are available in [`MODEL/`](MODEL/).
- **Notebook-first workflow:** explore and run the complete workflow in [`updated_CNN.ipynb`](updated_CNN.ipynb).

## Model at a glance

| Component | Configuration |
| --- | --- |
| Input | Grayscale image, `1 × 28 × 28` |
| Convolution block 1 | `1 → 32` channels, 3 × 3 convolution, batch normalization, ReLU, 2 × 2 max pool |
| Convolution block 2 | `32 → 64` channels, 3 × 3 convolution, batch normalization, ReLU, 2 × 2 max pool |
| Regularization | Dropout (`p=0.25`) before the classifier |
| Classifier | Linear layer, 1,600 features → 62 classes |
| Loss / optimizer | Cross-entropy / Adam (`lr=0.01`) |
| Training setup | Batch size 32; up to 5 epochs, with early stopping on validation loss |

## Get started

### Option 1: Run in Google Colab

Open [`updated_CNN.ipynb`](https://colab.research.google.com/github/Areeb-code/modular-CNN/blob/main/updated_CNN.ipynb) in Colab and run the cells in order. The notebook installs its dependencies and downloads EMNIST through `torchvision`.

The notebook configures a Weights & Biases logger. To use online experiment tracking, sign in to W&B in your environment; otherwise, adjust the logger configuration in the notebook.

### Option 2: Run locally

Clone the repository and install the core training dependencies in a Python environment:

```bash
git clone https://github.com/Areeb-code/modular-CNN.git
cd modular-CNN
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install torch torchvision pytorch-lightning torchmetrics wandb
```

Then launch Jupyter, open `updated_CNN.ipynb`, and run the cells. A CUDA-capable GPU is optional; PyTorch can also run on CPU. The notebook downloads the EMNIST ByClass dataset when its data-loading cell runs.

> **Note:** Dependencies are not currently pinned in a `requirements.txt` file. The notebook contains its own installation cells; the command above lists the core packages used by the training workflow.

## Use a TorchScript model

The repository includes TorchScript artifacts in [`MODEL/`](MODEL/). The following example loads one on CPU and checks its output shape:

```python
import torch

model = torch.jit.load("MODEL/modular_cnn_torchscript.pt", map_location="cpu")
model.eval()

# Replace this with a preprocessed grayscale character image.
# Expected shape: [batch_size, 1, 28, 28], with pixel values in [0, 1].
image = torch.zeros(1, 1, 28, 28)

with torch.inference_mode():
    logits = model(image)

print(logits.shape)                 # torch.Size([1, 62])
predicted_class = logits.argmax(1)  # class index from 0 to 61
```

This example verifies loading and inference shape; a blank image is **not** a meaningful recognition sample. For useful predictions, preprocess an input character consistently with the notebook and map the predicted class index to the EMNIST ByClass label order.

## Repository contents

```text
.
├── updated_CNN.ipynb      # Training, validation, export, and exploratory app code
├── MODEL/
│   ├── modular_cnn_torchscript.pt
│   └── modular_cnn_torchscript (1).pt
├── LICENSE
└── README.md
```

The notebook also contains exploratory FastAPI and Streamlit implementation steps. Those applications are generated from notebook cells and are **not currently included as standalone backend/frontend files** in the repository.

## Learning goals

- Understand how convolution and pooling build image features.
- Structure a model and training/validation steps with PyTorch Lightning.
- Work with a multi-class image dataset using `torchvision`.
- Save checkpoints and export a model for inference with TorchScript.

## Contributing

Ideas, bug reports, and improvements are welcome. Open an issue to discuss a change, or submit a pull request with a clear description of what you changed.

## License

This project is distributed under the [Apache License 2.0](LICENSE).

---

<div align="center">

Built as a practical way to learn the fundamentals of convolutional neural networks.

</div>
