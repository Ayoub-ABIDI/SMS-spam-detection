print("="*60)
print("TEST INSTALLATION")
print("="*60)

# PyTorch
print("\n[1] PyTorch:")
import torch
print(f"   Version: {torch.__version__}")
print(f"    PyTorch OK")

# TensorFlow
print("\n[2] TensorFlow:")
import tensorflow as tf
print(f"   Version: {tf.__version__}")
print(f"    TensorFlow OK")

# Transformers
print("\n[3] Transformers:")
import transformers
print(f"   Version: {transformers.__version__}")
print(f"    Transformers OK")

# Packages de base
print("\n[4] Packages de base:")
import pandas, numpy, matplotlib, seaborn, sklearn
print(f"    Tous OK")

# FAISS
print("\n[5] FAISS:")
import faiss
print(f"    FAISS (CPU) OK")

print("\n" + "="*60)
print("INSTALLATION COMPLÈTE!")
print("="*60)
print("\n Tu peux commencer le projet maintenant!")