import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from PIL import Image
from torchvision import transforms
from os.path import dirname
from bikit.utils import pil_loader
from pathlib import Path

class McdsDataset(Dataset):
    """PyTorch Dataset for MCDS. Multiclass-singlelabel dataset with 10 classes."""
    bikit_path = dirname(dirname(__file__))

    with open(Path(os.path.join(bikit_path, "data/datasets.json"))) as f:
        DATASETS = json.load(f)
    #name="mcds"

    def __init__(self,  name="mcds_Bikit", cache_dir=None, split="", transform=None,
                 load_all_in_mem=False, devel_mode=False):
        """

        :param name: Dataset name.
        :param split: Use 'trainval' or 'test', or '' (default) for all samples. Disclaimer: There are
                no original splits. The authors of bikit introduced this splits for comparability. Due to the dataset
                size it is recommended to do cross-validation on the trainval split.
        :param transform: Torch transformation for image data (this depends on your CNN).
        :param cache_dir: Path to cache_dir.
        :param load_all_in_mem: Whether or not to load all image data into memory (this depends on the dataset size and
            your memory). Loading all in memory can speed up your training.
        :param devel_mode: Only using 100 samples.
        """
        assert name in list(self.DATASETS.keys()), f"This name does not exists. Use something from {list(DATASETS.keys())}."
        self.name = name
        #assert split in ["", "trainval", "test"], f'You used split str({split}). Only ["", "trainval", "test"] are allowed.'
        bikit_path = Path(dirname(dirname(__file__)))
        self.csv_filename = Path(os.path.join(bikit_path, "data", self.name) + ".csv")

        # Misc
        self.split = split
        if cache_dir:
            self.cache_full_dir = Path(os.path.join(cache_dir))
        else:
            self.cache_full_dir = Path(os.path.join(os.path.expanduser("~"), ".bikit"))

        self.devel_mode = devel_mode
        self.class_names = self.DATASETS[name]["class_names"]
        self.num_classes = self.DATASETS[name]["num_classes"]
        self.load_all_in_mem = load_all_in_mem

        # Data prep
        self.transform = transform

        self.df = pd.read_csv(self.csv_filename)
        if split:
            self.df = self.df[self.df["split_bikit"] == split]
        if devel_mode:
            self.df = self.df[:100]
        self.n_samples = self.df.shape[0]

        if load_all_in_mem:
            assert NotImplemented

    def __getitem__(self, index):
        """Returns image as torch.Tensor and label as torch.Tensor with dimension (bs, num_classes)
        where 1 indicates that the label is present."""
        data = self.df.iloc[index]
        # Load and transform image
        img_filename = Path(os.path.join(self.cache_full_dir, data['img_path']))
        img = pil_loader(img_filename)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Get label with shape 10 or 8
        label = torch.FloatTensor(data[self.class_names].to_numpy().astype("float32"))
        print("LALALALALABEL")
        print(label)
        return img, label

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    print(__file__)

    print("===mcds_Bukhsh===")
    all_dataset = McdsDataset(name="mcds_Bukhsh", split="")
    trainval_dataset = McdsDataset(name="mcds_Bukhsh", split="trainval")
    test_dataset = McdsDataset(name="mcds_Bukhsh", split="test")
    development_dataset = McdsDataset(name="mcds_Bukhsh", split="test", devel_mode=True)

    img, targets = all_dataset[0]
    print(len(all_dataset))
    print(img.shape, targets.shape)
    print(len(all_dataset) )
    print(len(trainval_dataset) )
    print(len(test_dataset))
    print(len(development_dataset))

    print("===mcds_Bikit===")
    all_dataset = McdsDataset(split="")
    trainval_dataset = McdsDataset(split="trainval")
    test_dataset = McdsDataset(split="test")
    development_dataset = McdsDataset(split="test", devel_mode=True)

    img, targets = all_dataset[0]
    print(len(all_dataset))
    print(img.shape, targets.shape)
    print(list(targets.shape))
    print(targets.dtype)

    print(len(all_dataset))
    print(len(trainval_dataset))
    print(len(test_dataset))
    print(len(development_dataset))
    print("===Done===")

