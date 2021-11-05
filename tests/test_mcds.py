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

# Import module under test
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from bikit.datasets.mcds import McdsDataset

home_path = Path(path.expanduser('~'))
if home_path in [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]:
    image_path = home_path / ".bikit/mcds/Corrosion/no rust staining"
    makedirs(image_path)
    image_file = home_path / ".bikit/mcds/Corrosion/no rust staining/001_0fwtaowy.t1o.jpg"
    img_np = np.ones((92, 400, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)

def test_mcds_bukhsh():
    all_dataset = McdsDataset(name="mcds_Bukhsh", split="")
    trainval_dataset = McdsDataset(name="mcds_Bukhsh", split="trainval")
    test_dataset = McdsDataset(name="mcds_Bukhsh", split="test")
    development_dataset = McdsDataset(name="mcds_Bukhsh", split="test", devel_mode=True)
    cache_dir = McdsDataset(name="mcds_Bukhsh", split="", devel_mode=True, cache_dir=".bikit/mcds")
    transform = McdsDataset(name="mcds_Bukhsh", split="", devel_mode=True, transform=transforms.ToTensor())
    #all_in_mem = McdsDataset(name="mcds_Bukhsh", split="", load_all_in_mem=True, devel_mode=True)

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
    assert len(cache_dir) == 100
    assert len(transform) == 100
    #assert len(all_in_mem) == 100

def test_mcds_bikit():
    all_dataset = McdsDataset(split="")
    trainval_dataset = McdsDataset(split="trainval")
    test_dataset = McdsDataset(split="test")
    development_dataset = McdsDataset(split="test", devel_mode=True)
    cache_dir = McdsDataset(split="", devel_mode=True, cache_dir=".bikit/mcds")
    transform = McdsDataset(split="", devel_mode=True, transform=transforms.ToTensor())
    #all_in_mem = McdsDataset(split="", load_all_in_mem=True, devel_mode=True)

    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 92, 400]
    assert list(targets.shape) == [8]

    # Dataset length
    assert len(all_dataset) == 2597
    assert len(trainval_dataset) == 2147
    assert len(test_dataset) == 450
    assert len(development_dataset) == 100
    assert len(cache_dir) == 100
    assert len(transform) == 100
    #assert len(all_in_mem) == 100

def test_mcds_catch():
    with pytest.raises(Exception):
        d = McdsDataset(split="ERROR")
        d = McdsDataset(name="WRONG_NAME")

if __name__ == '__main__':
    test_mcds_bikit()
    test_mcds_bukhsh()
    test_mcds_catch()
