#!/usr/local/bin/python3

# Test Modules
import sys
import os
import pytest
from pathlib import Path
import shutil

# Import module under test
# sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from bikit.utils import list_datasets, download_dataset


def test_list_datasets():
    res = list_datasets(verbose=False)
    keys = list(res.keys())
    assert len(keys) >= 3
    assert res[keys[0]]


# @pytest.mark.skip(reason="no way of currently testing this")
def test_download_dataset():
    cache_dir = Path(os.path.expanduser('~/.bikit'))
    if os.path.exists(cache_dir / "test_rar"):
        shutil.rmtree(cache_dir / "test_rar")
    if os.path.exists(cache_dir / "test_zip"):
        shutil.rmtree(cache_dir / "test_zip")

    # download_dataset(name='demo_zip', cache_dir=cache_dir)
    download_dataset(name='test_rar', cache_dir=cache_dir)
    assert os.path.exists(cache_dir / "test_rar/test_rar.rar")
    assert os.path.exists(cache_dir / "test_rar/classification_dataset_balanced")

    download_dataset(name='test_zip', cache_dir=cache_dir)
    assert os.path.exists(cache_dir / "test_zip/test_zip.zip")
    assert os.path.exists(cache_dir / "test_zip/classification_dataset_balanced")


def test_force_redownload():
    cache_dir = Path(os.path.expanduser('~/.bikit'))
    if not os.path.exists(cache_dir / "test_zip/test_zip.zip"):
        os.mkdir(os.path.join(cache_dir, "test_zip"))
    os.mkdir(os.path.join(cache_dir, "test_zip", "force_test"))
    download_dataset(name="test_zip", force_redownload=True)
    assert not os.path.exists(os.path.join(cache_dir, "test_zip", "force_test"))


def test_zip_deletion():
    cache_dir = Path(os.path.expanduser('~/.bikit'))
    download_dataset(name="test_zip", force_redownload=True, rm_zip_or_rar=True)
    assert not os.path.exists(os.path.join(cache_dir, "test_zip", "test_zip.zip"))

def test_cache_dir():
    cache_dir = Path(os.path.expanduser('~/.bikit'))
    test_dir = os.path.join(cache_dir, "cache_test")
    if os.path.exists(test_dir):
        shutil.rmtree(os.path.join(test_dir))
    os.mkdir(test_dir)
    download_dataset(name="test_zip", cache_dir=test_dir)
    assert os.path.exists(os.path.join(test_dir, "test_zip", "test_zip.zip"))
    shutil.rmtree(os.path.join(test_dir))

if __name__ == '__main__':
    test_download_dataset()
    test_force_redownload()
    test_zip_deletion()
    test_cache_dir()
    print("===Done===")
