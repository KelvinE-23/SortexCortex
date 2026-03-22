# Waste Classification Baseline

This project provides a beginner-friendly PyTorch baseline for classifying waste images into 6 categories with transfer learning using ResNet18.

## Expected Dataset Layout

Store images in one folder per class:

```text
dataset/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

The folder names become the class labels automatically.

## Install

```bash
pip install torch torchvision pillow matplotlib scikit-learn numpy
```

## Train

```bash
python train.py --data-dir dataset --epochs 10 --batch-size 32 --freeze-backbone
```

Outputs are saved in `outputs/`:

- `outputs/checkpoints/best_model.pth`
- `outputs/checkpoints/last_model.pth`
- `outputs/training_history.json`
- `outputs/plots/confusion_matrix.png`

## Predict One Image

```bash
python predict.py --image-path sample.jpg --checkpoint outputs/checkpoints/best_model.pth
```

## Notes

- The script creates a reproducible train/validation split from a single dataset root.
- Validation evaluation includes a confusion matrix image.
- By default, training uses pretrained ResNet18 weights. The first run may download weights from torchvision.
