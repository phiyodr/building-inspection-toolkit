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
    image_path = home_path / ".cache/bikit/sdnet/D/CD/"
    makedirs(image_path)
    image_file = home_path / ".cache/bikit/sdnet/D/CD/7001-115.jpg"
    img_np = np.ones((256, 256, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)

@pytest.mark.parametrize("name", ["sdnet", "sdnet_binary"])
def test_sdnet_basic(name):
    all_dataset = BikitDataset(name, split="")
    transform_dataset = BikitDataset(name, split="",
                                     transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    train_dataset = BikitDataset(name, split="train")
    val_dataset = BikitDataset(name, split="val")
    test_dataset = BikitDataset(name, split="test")
    development_dataset = BikitDataset(name, split="test", devel_mode=True)
    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 256, 256]
    if name == "sdnet":
        assert list(targets.shape) == [6]
    elif name == "sdnet_binary":
        assert list(targets.shape) == [2]

    # Dataset length
    assert len(all_dataset) == 56092
    assert len(train_dataset) == 50488
    assert len(val_dataset) == 2808
    assert len(test_dataset) == 2796
    assert len(development_dataset) == 100
    assert len(transform_dataset) == 56092


@pytest.mark.skipif(home_path in travis_homes,
                    reason="Long-running test with real datasets for local use only, not on Travis.")
@pytest.mark.parametrize("name", ["sdnet", "sdnet_binary"])
def test_sdnet_local(name):

    # This test requieres at least 15GB of free RAM to work!
    #all_in_mem_dataset = BikitDataset(name="sdnet", split="", load_all_in_mem=True)
    all_in_mem_develmode = BikitDataset(name, split="", load_all_in_mem=True, devel_mode=True)


    #assert len(all_in_mem_dataset) == 56092
    assert len(all_in_mem_develmode) == 100

    #Test correct cache_dir func
    cache_test = BikitDataset(name, split="", cache_dir=Path(os.path.join(os.path.expanduser("~"), ".cache/bikit")))
    img, targets = cache_test[0]
    if name == "sdnet":
        assert list(targets.shape) == [6]
    elif name == "sdnet_binary":
        assert list(targets.shape) == [2]

if __name__ == '__main__':
    test_sdnet_local()
    test_sdnet_basic()