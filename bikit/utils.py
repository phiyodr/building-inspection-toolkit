import shutil
import urllib.request
import os
import hashlib
import sys
import zipfile
import json
import pprint
from os.path import dirname
from PIL import Image
from urllib.request import urlretrieve
from patoolib import extract_archive
from time import sleep
import requests
import ssl
import cv2
import requests
from io import BytesIO

from .models import DaclNet
import torch

pp = pprint.PrettyPrinter(indent=4)

bikit_path = dirname(__file__)

with open(os.path.join(bikit_path, "data/datasets.json")) as f:
    DATASETS = json.load(f)

DEMO_DATASETS = {"test_zip": {"description": "",
                              "download_name": "test_zip",
                              "license": "",
                              "urls": ["https://github.com/phiyodr/building-inspection-toolkit/raw/master/bikit/data/test_zip.zip"],
                              "original_names": ["test_zip.zip"],
                              "checksums": ["63b3722e69dcf7e14c879411c1907dae"],
                              "sizes": ["0.2 MB"]},
                 "test_rar": {"description": "",
                              "download_name": "test_rar",
                              "license": "",
                              "urls": ["https://github.com/phiyodr/building-inspection-toolkit/raw/master/bikit/data/test_rar.rar"],
                              "original_names": ["test_rar.rar"],
                              "checksums": ["c020266280b076ff123294ae51af2b11"],
                              "sizes": ["3.7 MB"]}}


def pil_loader(path):
    """Outputs an PIL Image object"""
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def cv2_loader(path):
    """Outputs an numpy.ndarray object"""
    # Can only use str not pathlib.PosixPath
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

def load_img_from_url(img_url):
    return Image.open(BytesIO(requests.get(img_url).content))


########## Model ##########

def list_models(verbose=True, cache_dir='~/.cache/bikit-models', force_redownload=False):
    """
    List all datasets available

    :param verbose: Print datasets
    :return: Return dictionary containing datasets name, url and original name.
    """
    models_metadata = get_metadata(cache_dir, force_redownload)
    if verbose:
        pp.pprint(models_metadata)
    return models_metadata

def download_model(name, cache_dir='~/.cache/bikit-models', force_redownload=False):
    models_metadata = get_metadata(cache_dir, force_redownload)
    all_model_names = list(models_metadata.keys())
    assert name in all_model_names, f"Please specify a valid model <name> out of {all_model_names}. You used {name}."
    base_url = "https://github.com/phiyodr/bikit-models/raw/master/models/"
    model_url = os.path.join(base_url, models_metadata[name]["pth_name"])
    filename = os.path.join(os.path.expanduser(cache_dir), models_metadata[name]["pth_name"])
    if not os.path.isfile(filename) or force_redownload:
        print(f"Start to download {name}.")
        urlretrieve(model_url, filename)
        print(f"Successfully downloaded model to {filename}.")
    else:
        print(f"Model {filename} already exists.")
    return filename

########## Metadata ##########


def download_metadata(cache_dir='~/.cache/bikit-models', force_redownload=False):
    """Download metadata.json from Repository."""
    cache_dir = os.path.expanduser(cache_dir)
    metadata_url = "https://github.com/phiyodr/bikit-models/raw/master/metadata.json"
    filename = os.path.join(cache_dir, "metadata.json")
    if not os.path.isfile(filename) or force_redownload:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        urlretrieve(metadata_url, filename)
        print(f"Successfully downloaded metadata.json to {filename}.")
    else:
        print(f"metadata.json already exists at {filename}.")

def read_metadata(cache_dir='~/.cache/bikit-models'):
    """Read metadata.json from directory."""
    filename = os.path.join(os.path.expanduser(cache_dir), "metadata.json")
    with open(filename) as json_file:
        metadata = json.load(json_file)
    return metadata

def get_metadata(cache_dir='~/.cache/bikit-models', force_redownload=False):
    "Return metadata.json as dict."
    filename = os.path.join(os.path.expanduser(cache_dir), "metadata.json")
    if not os.path.isfile(filename) or force_redownload:
        _ = download_metadata(cache_dir, force_redownload)
    return read_metadata(cache_dir)

def load_model(name, add_metadata=True, cache_dir="~/.cache/bikit-models", force_redownload=False):
    models_metadata = get_metadata(cache_dir, force_redownload)
    all_model_names = list(models_metadata.keys())
    assert name in all_model_names, f"Please specify a valid model <name> out of {all_model_names}. You used {name}."
    model_path = os.path.join(os.path.expanduser(cache_dir), models_metadata[name]["pth_name"])
    if not os.path.isfile(model_path) or force_redownload:
        download_model(name, cache_dir='~/.cache/bikit-models')

    cp = torch.load(model_path, map_location=torch.device('cpu'))
    model = DaclNet(base_name=cp['base'],
                    resolution = cp['resolution'],
                    hidden_layers=cp['hidden_layers'],
                    drop_prob=cp['drop_prob'],
                    num_class=cp['num_class'])
    model.load_state_dict(cp['state_dict'])
    model.eval()

    if add_metadata:
        metadata = get_metadata(cache_dir, force_redownload)[name]
        return model, metadata
    else:
        return model

########## Datasets ##########

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

def download_dataset(name, cache_dir='~/.cache/bikit', rm_zip_or_rar=False, force_redownload=False):
    """
    Download dataset if not on cache folder.

    :param name: Dataset name
    :param cache_dir: Cache directory
    :return:
    """
    if "test" in name:
        datasets = DEMO_DATASETS
        print(datasets[name])
    elif "meta" in name:
        print("Please download the needed datasets manually: [download_dataset(name) for name in ['bcd', 'codebrim-classif-balanced', 'mcds_Bikit', 'sdnet_binary']]")
        return 0
    else:
        datasets = DATASETS
        assert name in list(
            datasets.keys()), f"Please specify a valid <name> out of {list(datasets.keys())}. You used {name}."

    cache_dir = os.path.expanduser(cache_dir)

    # Check if cache exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Create cache {cache_dir}")
    else:
        print(f"{cache_dir} already exists")

    # Set defaults
    data_dict = datasets[name]
    download_name = data_dict["download_name"]
    if name in ["codebrim-classif-balanced"]:
        cache_full_dir = os.path.join(cache_dir, download_name, "classification_dataset_balanced")
    #elif name == "codebrim-classif-balanced":
    #    cache_full_dir = os.path.join(cache_dir, download_name, "classification_dataset")
    else:
        cache_full_dir = os.path.join(cache_dir, download_name)

    # remove old Data for a clean new Download
    if force_redownload:
        if os.path.exists(cache_full_dir):
            shutil.rmtree(cache_full_dir)
            print(f"The Folder {cache_full_dir} has been removed.")

    # cache_zip = os.path.join(cache_full_dir, data_dict['original_name'])
    urls = data_dict['urls']
    sizes = data_dict['sizes']
    file_type = data_dict['original_names'][0].split(".")[-1]
    checksums = data_dict['checksums']
    names = data_dict['original_names']
    #import pdb; pdb.set_trace()

    # Download if not available
    if not os.path.exists(cache_full_dir):
        print(f"Create folder {cache_full_dir}")
        os.makedirs(cache_full_dir)
        # Download

        # Some datasets consists of multiple files
        for idx, (url, file_name, checksum, size) in enumerate(zip(urls, names, checksums, sizes)):
            print(f"Start to download file {idx + 1} of {len(urls)} with {size}.")
            cache_zip = os.path.join(cache_full_dir, file_name)
            if name in ["codebrim-classif-balanced", "codebrim-classif", "sdnet_bikit", "sdnet_bikit_binary", "dacl1k"]:
                gdrive_download(total_size=size, download_id=url, full_cache_dir=cache_zip)
            else:
                if name == "sdnet":
                    ssl._create_default_https_context = ssl._create_unverified_context
                urllib.request.urlretrieve(url, filename=cache_zip, reporthook=_schedule)
            sleep(1)

            # Verify checksum
            if checksum:
                print("\nVerify checksum", end=" ")
                calculated_checksum = _md5(cache_zip)
                if calculated_checksum == checksum:
                    print("- checksum correct")
                else:
                    print(calculated_checksum, checksum)
                    print("- checksum wrong!")

            # Unzip/unrar
            if file_type == "zip":
                print("\nStart to unzip file", end=" ")
                with zipfile.ZipFile(cache_zip, 'r') as zip_ref:
                    if name in ["sdnet_bikit", "sdnet_bikit_binary"]:
                        cache_full_dir = "/".join(cache_full_dir.split("/")[:-1])
                    zip_ref.extractall(cache_full_dir)
                print("- unzip done!")
            elif file_type == "rar":
                print("Start to unraring file", end=" ")
                try:
                    extract_archive(cache_zip, outdir=cache_full_dir)
                except Exception as e:
                    print("\n", e)
                    print("\nHave you installed rar? Try <apt install unrar>.")
                    raise
                print("- unrar done!")

            # Rm zip/rar file
            if rm_zip_or_rar:
                print(f"Removing {cache_zip}.")
                os.remove(cache_zip)
    else:
        print(f"{cache_dir} and {cache_full_dir} already exists.")


def _progressbar(cur, total=100):
    """Source: https://www.programmersought.com/article/14355722336/"""
    percent = '{:.1%}'.format(cur / total)
    sys.stdout.write('\r')
    # sys.stdout.write("[%-50s] %s" % ('=' * int(math.floor(cur * 50 / total)),percent))
    sys.stdout.write("Download data [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()


def _schedule(blocknum, blocksize, totalsize):
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
    # print("download : %.2f%%" %(percent))
    _progressbar(percent)


def _md5(filename):
    # Source: https://stackoverflow.com/a/3431838
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def gdrive_download(total_size, download_id, full_cache_dir=""):
    """
    Download the codebrim_classif_dataset
    :param total_size: file size of the zipfile from the json file
    :param download_id: list of gdrive ids to download
    :param full_cache_dir: Cache directory
    """

    url = "https://docs.google.com/uc?export=download"

    # download the Zip file
    session = requests.Session()
    response = session.get(url, params={'id': download_id}, stream=True)
    token = _get_confirm_token(response)


    if token:
        params = {'id': download_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)

        _save_response_content(response=response, destination=full_cache_dir, totalsize=total_size)
        print(f"{full_cache_dir} is done")
    else:
        raise Exception("There was an Error while getting the download token!"
                        " This may occur when trying to download to often in a short time period."
                        " Please try again later!")


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value


def _save_response_content(response, destination, totalsize):
    chunk_size = 32768
    counter = 0
    print_counter = 0

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                counter += 1
                print_counter += 1
                filesize = float(totalsize.split(" ")[0]) * 1073741824
                _schedule(blocknum=counter, blocksize=chunk_size, totalsize=filesize)
    sys.stdout.write(f"\n{round((chunk_size * counter) / 1073741824, 2)} GB Downloaded. Download finished.\n", )


if __name__ == "__main__":
    name = "dacl1k"
    download_dataset(name)
    from bikit.datasets import BikitDataset  # Deprecated: from bikit.datasets.data import BikitDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    my_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_dataset = BikitDataset(name, split="train", transform=my_transform, return_type="pt")
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Use it in your training loop
    for i, (imgs, labels) in enumerate(train_dataset):
        print(i, imgs.shape, labels.shape)
        break

    test_data, test_meta = False, False
    if test_data:
        name = "codebrim-classif"

        #download_dataset(name, rm_zip_or_rar=True, force_redownload=False)
        print("===Download done===")
        from bikit.datasets import BikitDataset
        from torch.utils.data import DataLoader
        from torchvision import transforms

        my_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        trainval_dataset = BikitDataset(name, split="test", transform=my_transform)
        trainval_loader = DataLoader(dataset=trainval_dataset, batch_size=64, shuffle=False, num_workers=0)

        # Use it in your training loop
        for i, (imgs, labels) in enumerate(trainval_loader):
            print(i, imgs.shape, labels.shape, labels)
            if i > 1:
                break
        print("===Done===")
    elif test_meta:
        download_metadata()