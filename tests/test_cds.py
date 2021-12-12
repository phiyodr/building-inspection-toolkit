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
from bikit.datasets.cds import CdsDataset

home_path = Path(path.expanduser('~'))
travis_homes = [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]

if home_path in travis_homes:
    image_path = home_path / ".bikit/bcd/"
    makedirs(image_path)
    image_file = home_path / ".bikit/bcd/1.jpg"
    img_np = np.ones((224, 224, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)


def test_cds_basic():
    all_dataset = CdsDataset(split="")
    transform_dataset = CdsDataset(split="",
                                  transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    train_dataset = CdsDataset(split="train")
    val_dataset = CdsDataset(split="val")
    test_dataset = CdsDataset(split="test")
    development_dataset = CdsDataset(split="test", devel_mode=True)
    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 299, 299]
    assert list(targets.shape) == [2]

    # Dataset length
    assert len(all_dataset) == 1028
    assert len(train_dataset) == 824
    assert len(val_dataset) == 104
    assert len(test_dataset) == 100
    assert len(development_dataset) == 100
    assert len(transform_dataset) == 1028

@pytest.mark.skipif(home_path in travis_homes,
                    reason="Long-running test with real datasets for local use only, not on Travis.")
def test_cds_local():
    all_in_mem_dataset = CdsDataset(split="", load_all_in_mem=True)
    all_in_mem_develmode = CdsDataset(split="", load_all_in_mem=True, devel_mode=True)

    assert len(all_in_mem_dataset) == 1028
    assert len(all_in_mem_develmode) == 100

    #Test correct cache_dir func
    cache_test = CdsDataset(split="", cache_dir=Path(os.path.join(os.path.expanduser("~"), ".bikit")))
    img, targets = cache_test[0]
    assert list(targets.shape) == [2]

if __name__ == '__main__':
    test_cds_local()
    test_cds_basic()