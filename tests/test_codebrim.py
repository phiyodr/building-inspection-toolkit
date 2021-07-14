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
from bikit.datasets.codebrim import CodebrimDataset

home_path = Path(path.expanduser('~'))

if home_path in [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]:
    image_path = home_path / ".bikit/codebrim-classif-balanced/classification_dataset_balanced/train/background/"
    makedirs(image_path)
    image_file = home_path / ".bikit/codebrim-classif-balanced/classification_dataset_balanced/train/background/image_0000001_crop_0000001.png"
    img_np = np.ones((379, 513, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)

def test_codebrim():
    all_dataset = CodebrimDataset(split_type="")
    train_dataset = CodebrimDataset(split_type="train")
    val_dataset = CodebrimDataset(split_type="val")
    test_dataset = CodebrimDataset(split_type="test")
    development_dataset = CodebrimDataset(split_type="test", devel_mode=True)
    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 379, 513]
    assert list(targets.shape) == [6]

    # Dataset length
    assert len(all_dataset) == 7261
    assert len(train_dataset) == 6013
    assert len(val_dataset) == 616
    assert len(test_dataset) == 632
    assert len(development_dataset) == 100


if __name__ == '__main__':
    test_codebrim()
