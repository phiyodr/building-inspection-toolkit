#!/usr/local/bin/python3

# Test Modules
import sys
import pytest
from os import path, makedirs
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
import os

# Import module under test
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from bikit.datasets.data import BikitDataset

home_path = Path(path.expanduser('~'))
travis_homes = [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]

if home_path in travis_homes:
    image_path = home_path / ".bikit/mcds/Corrosion/no rust staining"
    makedirs(image_path)
    image_file = home_path / ".bikit/mcds/Corrosion/no rust staining/001_0fwtaowy.t1o.jpg"
    img_np = np.ones((92, 400, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)


def test_mcds_bukhsh_basic():
    all_dataset = BikitDataset(name="mcds_Bukhsh",split="")
    trainval_dataset = BikitDataset(name="mcds_Bukhsh",  split="trainval")
    test_dataset = BikitDataset(name="mcds_Bukhsh", split="test")
    development_dataset = BikitDataset(name="mcds_Bukhsh", split="test", devel_mode=True)
    transform_dataset = BikitDataset(name="mcds_Bukhsh", split="", devel_mode=True,
                                    transform=transforms.Compose(
                                        [transforms.Resize((256, 256)), transforms.ToTensor()]))

    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 92, 400]
    assert list(targets.shape) == [10]

    # Dataset length
    assert len(all_dataset) == 2612
    assert len(trainval_dataset) == 2114
    assert len(test_dataset) == 498
    assert len(development_dataset) == 100
    assert len(transform_dataset) == 100


@pytest.mark.skipif(home_path in travis_homes,
                    reason="Long-running test with real datasets for local use only, not on Travis.")
def test_mcds_bukhsh_local():
    all_in_mem = BikitDataset(name="mcds_Bukhsh", split="", load_all_in_mem=True)
    all_in_mem_develmode = BikitDataset(name="mcds_Bukhsh", split="", load_all_in_mem=True, devel_mode=True)

    assert len(all_in_mem) == 2612
    assert len(all_in_mem_develmode) == 100

    #Test correct cache_dir func
    cache_test = BikitDataset(name="mcds_Bukhsh", split="", cache_dir=Path(os.path.join(os.path.expanduser("~"), ".bikit")))
    img, targets = cache_test[0]
    assert list(targets.shape) == [10]


def test_mcds_bikit_basic():
    all_dataset = BikitDataset(name="mcds_Bikit", split="")
    train_dataset = BikitDataset(name="mcds_Bikit", split="train")
    valid_dataset = BikitDataset(name="mcds_Bikit", split="valid")
    test_dataset = BikitDataset(name="mcds_Bikit", split="test")
    development_dataset = BikitDataset(name="mcds_Bikit", split="test", devel_mode=True)
    transform_dataset = BikitDataset(name="mcds_Bikit", split="",
                                    transform=transforms.Compose(
                                        [transforms.Resize((256, 256)), transforms.ToTensor()]))

    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 92, 400]
    assert list(targets.shape) == [8]

    # Dataset length
    assert len(all_dataset) == 2597
    assert len(train_dataset) == 2057
    assert len(valid_dataset) == 270
    assert len(test_dataset) == 270
    assert len(development_dataset) == 100
    assert len(transform_dataset) == 2597


@pytest.mark.skipif(home_path in travis_homes,
                    reason="Long-running test with real datasets for local use only, not on Travis.")
def test_mcds_bikit_local():
    all_in_mem = BikitDataset(name="mcds_Bikit", split="", load_all_in_mem=True)
    all_in_mem_develmode = BikitDataset(name="mcds_Bikit", split="", load_all_in_mem=True, devel_mode=True)

    assert len(all_in_mem_develmode) == 100
    assert len(all_in_mem) == 2597

    #Test correct cache_dir func
    cache_test = BikitDataset(name="mcds_Bikit", split="", cache_dir=Path(os.path.join(os.path.expanduser("~"), ".bikit")))
    img, targets = cache_test[0]
    assert list(targets.shape) == [8]


def test_mcds_catch():
    with pytest.raises(Exception):
        d = BikitDataset(name="mcds_Bikit", split="ERROR")
        d = BikitDataset(name="WRONG_NAME")


if __name__ == '__main__':
    test_mcds_bikit_local()
    test_mcds_bikit_basic()
    test_mcds_bukhsh_local()
    test_mcds_bukhsh_basic()

    test_mcds_catch()
