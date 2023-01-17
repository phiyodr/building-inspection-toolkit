import gdown
import os
import hashlib
import zipfile
import json
import pprint
from os.path import dirname
from PIL import Image
from urllib.request import urlretrieve
from time import sleep
import requests
import cv2
import requests
from io import BytesIO


pp = pprint.PrettyPrinter(indent=4)

bikit_path = dirname(__file__)

with open(os.path.join(bikit_path, "data/datasets.json")) as f:
    DATASETS = json.load(f)

DEMO_DATASETS = {"test_zip": {"description": "",
                              "download_name": "test_zip",
                              "license": "",
                              "urls": ["https://github.com/phiyodr/building-inspection-toolkit/raw/master/bikit/data/test_zip.zip"],
                              "original_names": ["test_zip.zip"],
                              "checksums": ["63b3722e69dcf7e14c879411c1907dae"]}}


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
    from .models import DaclNet

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


def download_dataset(name, cache_dir='~/.cache/bikit', rm_zip_or_rar=True, force_redownload=False):
    # Get details from DATASETS
    dct = DATASETS[name]
    uid = dct["url"]
    print(f"The {name} dataset was published at {dct['publications']}\n",
        f"With downloading you accept the licence: {dct['license']}\n",
        f"More details at {dct['webpage']}")

    cache_full_dir = os.path.expanduser(cache_dir)
    zip_file = f"{dct['download_name']}.zip"
    cache_zip_file = os.path.join(cache_full_dir, zip_file)
    cache_zip_folder = os.path.splitext(cache_zip_file)[0]

    # Create cache directory 
    if not os.path.exists(cache_full_dir):
        print(f"Create folder {cache_full_dir}")
        os.makedirs(cache_full_dir)
    
    # Download if not already present or not forced
    if not os.path.exists(cache_zip_folder) or force_redownload:
        # Download
        url = f'https://drive.google.com/uc?id={uid}'
        gdown.download(url, cache_zip_file, quiet=False)

        # Unzip
        print("\nStart to unzip file", end=" ")
        with zipfile.ZipFile(cache_zip_file, 'r') as zip_ref:
            zip_ref.extractall(cache_full_dir)
        print("- unzip done!")

        if rm_zip_or_rar:
            print(f"Removing {cache_zip_file}")
            os.remove(cache_zip_file)

    else:
        print(f"Folder {cache_zip_folder} already exists.\n",
            f"Use argument set 'force_redownload=True' to force redownload.")

if __name__ == "__main__":
    name = "dacl1k"
    download_dataset(name)
    from bikit.datasets import BikitDataset  # Deprecated: from bikit.datasets.data import BikitDataset
    from torch.utils.data import DataLoader
    from torchvision import transforms
    my_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_dataset = BikitDataset(name, split="train", transform=my_transform, return_type="pt")
    Train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)

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