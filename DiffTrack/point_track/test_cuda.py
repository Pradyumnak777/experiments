import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
import tensorflow as tf
print(f"TensorFlow CUDA available: {tf.config.list_physical_devices('GPU')}")
