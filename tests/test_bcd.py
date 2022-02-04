#!/usr/local/bin/python3

# Test Modules
import sys
import pytest
from torchvision import transforms
from os import path, makedirs
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import os

# Import module under test
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from bikit.datasets import BikitDataset

home_path = Path(path.expanduser('~'))
travis_homes = [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]

if home_path in travis_homes:
    image_path = home_path / ".cache/bikit/bcd/"
    makedirs(image_path)
    image_file = home_path / ".cache/bikit/bcd/1.jpg"
    img_np = np.ones((224, 224, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)


def test_bcd_basic():
    all_dataset = BikitDataset(name="bcd", split="")
    transform_dataset = BikitDataset(name="bcd", split="",
                                  transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    train_dataset = BikitDataset(name="bcd", split="train")
    val_dataset = BikitDataset(name="bcd", split="val")
    test_dataset = BikitDataset(name="bcd", split="test")
    development_dataset = BikitDataset(name="bcd", split="test", devel_mode=True)
    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 224, 224]
    assert list(targets.shape) == [2]

    # Dataset length
    assert len(all_dataset) == 6069
    assert len(train_dataset) == 4869
    assert len(val_dataset) == 600
    assert len(test_dataset) == 600
    assert len(development_dataset) == 100
    assert len(transform_dataset) == 6069

@pytest.mark.skipif(home_path in travis_homes,
                    reason="Long-running test with real datasets for local use only, not on Travis.")
def test_bcd_local():
    all_in_mem_dataset = BikitDataset(name="bcd", split="", load_all_in_mem=True)
    all_in_mem_develmode = BikitDataset(name="bcd", split="", load_all_in_mem=True, devel_mode=True)

    assert len(all_in_mem_dataset) == 6069
    assert len(all_in_mem_develmode) == 100

    #Test correct cache_dir func
    cache_test = BikitDataset(name="bcd", split="", cache_dir=Path(os.path.join(os.path.expanduser("~"), ".cache/bikit")))
    img, targets = cache_test[0]
    assert list(targets.shape) == [2]

if __name__ == '__main__':
    test_bcd_local()
    test_bcd_basic()
