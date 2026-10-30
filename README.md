# Food-101 Classification: Fine-Tuned ViT vs. Zero-Shot CLIP

This project compares two approaches to food image classification on the **Food-101** dataset: fine-tuning a Vision Transformer (ViT) for 101 food categories and evaluating a pretrained CLIP model using zero-shot classification.

## Project Overview

The notebook explores whether a general-purpose vision-language model can classify food images without task-specific training, compared with a vision model fine-tuned on the target dataset.

### Approach 1: Fine-tuned Vision Transformer (ViT)

- Dataset: Food-101 training and test splits from `torchvision.datasets.Food101`.
- Starting checkpoint: [`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k).
- Replaces the classification head for **101 classes**.
- Uses Hugging Face Transformers `Trainer` to train for 3 epochs, with a per-device training batch size of 32.
- Evaluates classification accuracy on the test split.

### Approach 2: Zero-Shot CLIP

- Model: [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32).
- Uses the pretrained model without fine-tuning it on Food-101.
- Supplies the 101 Food-101 class names as candidate labels, with the prompt template: `This is a photo of a {}.`
- Compares each predicted top label with the ground-truth class.

**Recorded CLIP test accuracy: 81.20%** (approximately), based on the notebook output. The fine-tuned ViT achieved 89.36% test accuracy.

## Dataset

Food-101 contains **101 food categories**, with 1,000 images per category (750 training images and 250 test images per class in the standard split).

The notebook downloads the dataset using `torchvision.datasets.Food101`. The CLIP evaluation reads the test image list from `data/food-101/meta/test.txt` and constructs image paths under `data/food-101/images/`.

## Requirements

The notebook uses Python and these main libraries:

- PyTorch
- torchvision
- Transformers
- Pillow

Install the dependencies in your Python environment:

```bash
pip install torch torchvision transformers pillow
```

Depending on your hardware and installed PyTorch build, you may need a compatible CUDA-enabled PyTorch installation to use an NVIDIA GPU.

## How to Run

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Open `fine_tune_vit_vs_clip.ipynb` in Jupyter Notebook or VS Code.
4. Run the notebook cells in order.

The first run downloads the Food-101 dataset and pretrained model weights. Ensure that you have enough disk space, GPU memory (if using a GPU), and time for model training and test-set inference.

## Results

| Method | Model | Training on Food-101 | Recorded test accuracy |
|---|---|---:|---:|
| Fine-tuned ViT | `google/vit-base-patch16-224-in21k` | Yes, 3 epochs configured | **89.36%** |
| Zero-shot CLIP | `openai/clip-vit-base-patch32` | No | **81.20%** |

The fine-tuned ViT achieved 89.36% test accuracy, compared with 81.20% for zero-shot CLIP. This result illustrates the benefit of task-specific fine-tuning for this Food-101 classification task, while CLIP provides strong performance without training on the target dataset.

## Repository Contents

- `fine_tune_vit_vs_clip.ipynb` — dataset loading, ViT fine-tuning, and zero-shot CLIP evaluation.

## Notes

- The ViT training configuration enables `bf16=True`. This requires hardware and a PyTorch setup that support bfloat16 training; change this setting if your device does not support it.
- The notebook's CLIP evaluation processes the test image paths in batches of 32.
- Results can vary with software versions, hardware, training settings, and whether the notebook is rerun from scratch.
