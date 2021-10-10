#!/usr/local/bin/python3

# Test Modules
import sys
import pytest
from os import path, makedirs
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Import module under test
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from bikit.datasets.bcd import BcdDataset

home_path = Path(path.expanduser('~'))

if home_path in [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]:
    image_path = home_path / ".bikit/bcd/"
    makedirs(image_path)
    image_file = home_path / ".bikit/bcd/1.jpg"
    img_np = np.ones((224, 224, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)

def test_codebrim():
    all_dataset = BcdDataset(split="")
    all_in_mem_dataset = BcdDataset(split="", load_all_in_mem=True)
    train_dataset = BcdDataset(split="train")
    val_dataset = BcdDataset(split="val")
    test_dataset = BcdDataset(split="test")
    development_dataset = BcdDataset(split="test", devel_mode=True)
    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 224, 224]
    assert list(targets.shape) == [1]

    # Dataset length
    assert len(all_dataset) == 6069
    assert len(all_in_mem_dataset) == 6069
    assert len(train_dataset) == 4869
    assert len(val_dataset) == 600
    assert len(test_dataset) == 600
    assert len(development_dataset) == 100


if __name__ == '__main__':
    test_codebrim()
