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
from bikit.datasets.mcds import McdsDataset

home_path = Path(path.expanduser('~'))
if home_path in [Path("/home/travis"), Path("C:/Users/travis"), Path("/Users/travis")]:
    image_path = home_path / ".bikit/mcds/ExposedReinforcement/no exposed reinforcement"
    makedirs(image_path)
    image_file = home_path / ".bikit/mcds/ExposedReinforcement/no exposed reinforcement/FL-580075-FLD-DC-VI-011-103-DSCF2194_44s5dewc.st1.jpg"
    img_np = np.ones((297, 615, 3), dtype=np.int8) * 100
    img_pil = Image.fromarray(np.uint8(img_np)).convert('RGB')
    img_pil.save(image_file)

def test_mcds_bikit():
    all_dataset = McdsDataset(split_type="")
    trainval_dataset = McdsDataset(split_type="trainval_bikit")
    test_dataset = McdsDataset(split_type="test_bikit")
    development_dataset = McdsDataset(split_type="test_bikit", devel_mode=True)
    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 297, 615]
    assert list(targets.shape) == [10]

    # Dataset length
    assert len(all_dataset) == 2612
    assert len(trainval_dataset) == 1844
    assert len(test_dataset) == 768
    assert len(development_dataset) == 100

def test_mcds_bukhsh():
    all_dataset = McdsDataset(name="mcds_Bukhsh", split_type="")
    trainval_dataset = McdsDataset(name="mcds_Bukhsh", split_type="trainval_bikit")
    test_dataset = McdsDataset(name="mcds_Bukhsh", split_type="test_bikit")
    development_dataset = McdsDataset(name="mcds_Bukhsh", split_type="test_bikit", devel_mode=True)

    img, targets = all_dataset[0]
    assert img.dtype == torch.float32
    assert targets.dtype == torch.float32
    assert list(img.shape) == [3, 297, 615]
    assert list(targets.shape) == [10]

    # Dataset length
    assert len(all_dataset) == 2612
    assert len(trainval_dataset) == 1844
    assert len(test_dataset) == 768
    assert len(development_dataset) == 100

def test_mcds_catch():
    with pytest.raises(Exception):
        d = McdsDataset(split_type="ERROR")
        d = McdsDataset(name="WRONG_NAME")

if __name__ == '__main__':
    test_mcds()
