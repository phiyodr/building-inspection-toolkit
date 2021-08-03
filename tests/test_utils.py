#!/usr/local/bin/python3

# Test Modules
import sys
import os
import pytest
from pathlib import Path

# Import module under test
#sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from bikit.utils import list_datasets, download_dataset


def test_list_datasets():
    res = list_datasets(verbose=False)
    keys = list(res.keys())
    assert len(keys) >= 3
    assert res[keys[0]]


#@pytest.mark.skip(reason="no way of currently testing this")
def test_download_dataset():
    cache_dir = Path(os.path.expanduser('~/.bikit'))
    #download_dataset(name='demo_zip', cache_dir=cache_dir)
    download_dataset(name='demo_rar', cache_dir=cache_dir)
    assert os.path.exists(cache_dir / "test_rar/test_rar.rar")
    assert os.path.exists(cache_dir / "test_rar/multi_classifier_data")

    download_dataset(name='demo_rar', cache_dir=cache_dir)
    assert os.path.exists(cache_dir / "test_zip/test_zip.zip")
    assert os.path.exists(cache_dir / "test_zip/classification_dataset_balanced")
    print("===Done===")