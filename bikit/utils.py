import urllib.request
import os
import hashlib
import sys
import zipfile
import json
import pprint
from os.path import dirname
from PIL import Image
from patoolib import extract_archive
from time import sleep

pp = pprint.PrettyPrinter(indent=4)

bikit_path = dirname(__file__)
with open(os.path.join(bikit_path, "data/datasets.json")) as f:
    DATASETS = json.load(f)

DEMO_DATASETS = {"demo_zip": {"description":  "",
                 "download_name": "demo_zip",
                 "url" : "tudo",
                 "original_name": "demo_zip.zip",
                 "checksum": "todo",
                 "size": "7.9 GB"},
  "demo_rar": {"description":  "",
                 "download_name": "rar_demo",
                 "url" : "https://github.com/phiyodr/bridge-inspection-toolkit/raw/master/bikit/data/demo_rar.rar",
                 "original_name": "demo_rar.rar",
                 "checksum": "63b3722e69dcf7e14c879411c1907dae",
                 "size": "3.7 MB"}}



def pil_loader(path) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def list_datasets(verbose=True):
    """
    List all datasets available

    :param verbose: Print datasets
    :return: Return dictionary containing datasets name, url and original name.
    """
    datasets = DATASETS
    if verbose:
        pp.pprint(datasets)
    return datasets


def download_dataset(name, cache_dir='~/.bikit'):
    """
    Download dataset if not on cache folder.

    :param name: Dataset name
    :param cache_dir: Cache directory
    :return:
    """
    if "demo" in name:
        DATASETS = DEMO_DATASETS
        print(DATASETS[name])
    else:
        assert name in list(DATASETS.keys()), f"Please specify a valid <name> out of {list(DATASETS.keys())}. You used {name}."
    # Check if cache exist
    cache_dir = os.path.expanduser(cache_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Create cache {cache_dir}")
    else:
        print(f"{cache_dir} already exists")

    # Set defaults
    data_dict = DATASETS[name]
    download_name = data_dict["download_name"]
    cache_full_dir = os.path.join(cache_dir, download_name)
    cache_zip = os.path.join(cache_full_dir, data_dict['original_name'])
    url = data_dict['url']
    size = data_dict['size']
    file_type = data_dict['original_name'].split(".")[-1]
    checksum = data_dict['checksum']

    # Download if not available
    if not os.path.exists(cache_full_dir):
        print(f"Create folder {cache_full_dir}")
        os.makedirs(cache_full_dir)
        # Download
        print(f"Start to download {size} of data")
        urllib.request.urlretrieve(url, cache_zip, _schedule)
        sleep(1)
        print("\nDownload done!")
        # Unzip/unrar
        if file_type == "zip":
            print("Start to unzip file")
            with zipfile.ZipFile(cache_zip, 'r') as zip_ref:
                zip_ref.extractall(cache_full_dir)
            print("Unzip done!")
        if file_type == "rar":
            print("Start to unraring file")
            try:
                extract_archive(cache_zip, outdir=cache_full_dir)
            except Exception as e:
                print(e)
                print("Have you installed rar? Try <apt install unrar>.")
                raise
            print("Unrar done!")
    else:
        print(f"{cache_dir} and {cache_full_dir} already exists")

    # Verify
    print("Verify file")
    if checksum:
        calculated_checksum = _md5(cache_zip)
        if calculated_checksum == checksum:
            print("Checksum correct")
        else:
            print(calculated_checksum, checksum)
    print("Done!")


def _progressbar(cur, total=100):
    """Source: https://www.programmersought.com/article/14355722336/"""
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    # sys.stdout.write("[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)),percent))
    sys.stdout.write("Download data [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()


def _schedule(blocknum,blocksize,totalsize):
    """
    Source: https://www.programmersought.com/article/14355722336/

    blocknum: currently downloaded block
    blocksize: block size for each transfer
    totalsize: total size of web page files
    """
    if totalsize == 0:
        percent = 0
    else:
        percent = blocknum * blocksize / totalsize
    if percent > 1.0:
        percent = 1.0
    percent = percent * 100
    #print("download : %.2f%%" %(percent))
    _progressbar(percent)


def _md5(filename):
    # Source: https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

if __name__ == "__main__":
    list_datasets(verbose=True)
    #download_dataset(name='codebrim-classif', cache_dir='/home/philipp/.bikit')
    download_dataset(name='mcds', cache_dir='~/.bikit')
    print("===Done===")