import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import os
import PIL
from PIL import Image
from torchvision import transforms
from os.path import dirname
from bikit.utils import pil_loader, cv2_loader, DATASETS
from pathlib import Path
from tqdm import tqdm


class BikitDataset(Dataset):
    """PyTorch Dataset for all Datasets"""
    #bikit_path = Path(dirname(dirname(__file__)))
    #bikit_path = Path(os.path.join(bikit_path, "bikit"))
    bikit_path = Path(os.path.join(dirname(dirname(__file__)), "bikit"))

    with open(Path(os.path.join(bikit_path, "data/datasets.json"))) as f:
        DATASETS = json.load(f)

    def __init__(self, name, split=None, cache_dir=None, transform=None, img_type="pil", return_type="pt",
                 load_all_in_mem=False, devel_mode=False):
        """

        :param name: Dataset name.
        :param split: Use 'train', 'val' or 'test.
        :param transform: Torch transformation for image data (this depends on your CNN).
        :param img_type: Load image as PIL or CV2.
        :param return_type: Returns Torch tensor ('pt') or numpy ('np').
        :param cache_dir: Path to cache_dir.
        :param load_all_in_mem: Whether or not to load all image data into memory (this depends on the dataset size and
            your memory). Loading all in memory can speed up your training.
        :param devel_mode:
        """
        assert img_type.lower() in ["pil", "cv2"], f"Not a valid imgage type. Use something from ['pil','cv2']."
        if img_type == "pil":
            self.img_loader = pil_loader
        elif img_type == "cv2":
            self.img_loader = cv2_loader
        assert name in list(self.DATASETS.keys()), f"This name does not exists. Use something from {list(DATASETS.keys())}."
        bikit_path = Path(os.path.join(dirname(dirname(__file__)), "bikit"))

        self.csv_filename = Path(os.path.join(bikit_path, "data", name) + ".csv")
        self.split_column =DATASETS[name]["split_column"]
        self.available_splits =DATASETS[name]["splits"]
        assert split in self.available_splits + [""], f"{split} is not a valid split. Use somethong from {self.available_splits}."
        assert return_type in ["pt", "np"],  f"{return_type} is not a valid return_type. Use somethong from {['pt', 'np']}."
        self.return_type = return_type

        # Misc
        self.split = split
        if cache_dir:
            self.cache_full_dir = Path(os.path.join(cache_dir))
        else:
            self.cache_full_dir = Path(os.path.join(os.path.expanduser("~"), ".cache/bikit"))

        self.devel_mode = devel_mode
        self.class_names = self.DATASETS[name]["class_names"]
        self.num_classes = self.DATASETS[name]["num_classes"]
        self.load_all_in_mem = load_all_in_mem

        # Data prep
        self.transform = transform
        self.df = pd.read_csv(self.csv_filename)
        if split:
            self.df = self.df[self.df[self.split_column] == split]
        if devel_mode:
            self.df = self.df[:100]
        self.n_samples = self.df.shape[0]

        if load_all_in_mem:
            self.img_dict = {}
            for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0], desc="Load images in CPU memory"):
                img_filename = Path(os.path.join(self.cache_full_dir, row['img_path']))
                img_name = row['img_name']
                img = self.img_loader(img_filename)
                self.img_dict[img_name] = img

    def __getitem__(self, index):
        """Returns image as torch.Tensor and label as torch.Tensor with dimension (bs, num_classes)
        where 1 indicates that the label is present."""
        data = self.df.iloc[index]

        # Get image
        if self.load_all_in_mem:
            img = self.img_dict[data['img_name']]
        else:
            img_filename = Path(os.path.join(self.cache_full_dir, data['img_path']))
            img = self.img_loader(img_filename)
        if self.transform:
            img = self.transform(img)
        elif self.return_type == "pt":
            img = transforms.ToTensor()(img)
        if (self.return_type == "np") and isinstance(img, PIL.Image.Image):
            img = np.array(img)
        # Get label with shape (1,)
        if self.return_type == "np":
            label = data[self.class_names].to_numpy().astype("float32")
        else:
            label = torch.FloatTensor(data[self.class_names].to_numpy().astype("float32"))
        return img, label

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    print(__file__)
    dataset1 = BikitDataset(name="cds", split="train")
    dataset2 = BikitDataset(name="sdnet", split="train", img_type="cv2")
    dataset3 = BikitDataset(name="bcd", split="train", img_type="cv2")
    dataset4 = BikitDataset(name="mcds_Bikit", split="train", img_type="cv2")
    #train_dataset = BikitDataset(split="", load_all_in_mem=True)
    img, targets = dataset1[0]
    print(img.shape, targets.shape)
    print(len(dataset1))
    print("======")
    for key in DATASETS:
        if key not in ["codebrim-classif"]:
            for split in DATASETS[key]["splits"]:
                for img_type in ["pil", "cv2"]:
                    for return_type in ["pt", "np"]:
                        dataset = BikitDataset(name=key, split=split, img_type=img_type, return_type=return_type)
                        img, targets = dataset[0]
                        print(key, split, img_type, return_type, img.shape, type(img), targets.shape, type(targets))