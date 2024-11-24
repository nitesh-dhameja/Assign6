import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import MNIST_CNN
import pytest

def test_model_parameters():
    model = MNIST_CNN()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 20000, f"Model has {param_count} parameters, should be less than 20000"

def test_model_output_shape():
    model = MNIST_CNN()
    batch_size = 64
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    assert output.shape == (batch_size, 10), f"Expected output shape (64, 10), got {output.shape}"

def test_model_forward_pass():
    model = MNIST_CNN()
    x = torch.randn(1, 1, 28, 28)
    try:
        output = model(x)
        assert output.shape == (1, 10)
    except Exception as e:
        pytest.fail(f"Forward pass failed with error: {str(e)}") 