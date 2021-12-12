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
from bikit.datasets.codebrim import CodebrimDataset

home_path = Path(path.expanduser('~'))
travis_homes = [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]

if home_path in travis_homes:
    image_path = home_path / ".bikit/codebrim-classif-balanced/classification_dataset_balanced/train/background/"
    Path(image_path).mkdir(parents=True, exist_ok=True)
    image_file = home_path / ".bikit/codebrim-classif-balanced/classification_dataset_balanced/train/background/image_0000001_crop_0000001.png"
    img_np = np.ones((379, 513, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)


def test_codebrim_basic():
    all_dataset = CodebrimDataset(split="")
    train_dataset = CodebrimDataset(split="train")
    val_dataset = CodebrimDataset(split="val")
    test_dataset = CodebrimDataset(split="test")
    development_dataset = CodebrimDataset(split="test", devel_mode=True)
    transform_dataset = CodebrimDataset(split="",
                                        transform=transforms.Compose(
                                            [transforms.Resize((256, 256)), transforms.ToTensor()]))
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
    assert len(transform_dataset) == 7261


@pytest.mark.skipif(home_path in travis_homes,
                    reason="Long-running test with real datasets for local use only, not on Travis.")
def test_codebrim_local():

    # all_in_mem Test requires at least 10GB of free RAM to work
    # all_in_mem = CodebrimDataset(split="", load_all_in_mem=True)
    all_in_mem_develmode = CodebrimDataset(split="", load_all_in_mem=True, devel_mode=True)

    #Test correct cache_dir func
    cache_test = CodebrimDataset(split="", cache_dir=Path(os.path.join(os.path.expanduser("~"), ".bikit")))
    img, targets = cache_test[0]
    assert list(targets.shape) == [6]

    # assert len(all_in_mem) == 7261
    assert len(all_in_mem_develmode) == 100

if __name__ == '__main__':
    test_codebrim_local()
    test_codebrim_basic()
