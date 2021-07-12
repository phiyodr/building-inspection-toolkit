import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from PIL import Image
from torchvision import transforms
from os.path import dirname
from bikit.utils import pil_loader


#parent_dir = os.path.abspath(".")
#print(parent_dir)
#with open(os.path.join(parent_dir, "bikit/data/datasets.json")) as f:
#    DATASETS = json.load(f)

#with open("../data/datasets.json") as f:
#    DATASETS = json.load(f)
#ROOT_DIR = dirname(os.path.abspath(__file__))
#print("ROOT_DIR", ROOT_DIR)



class McdsDataset(Dataset):
    """PyTorch Dataset for MCDS. Multiclass-singlelabel dataset with 10 classes."""
    bikit_path = dirname(dirname(__file__))

    with open(os.path.join(bikit_path, "data/datasets.json")) as f:
        DATASETS = json.load(f)
    #name="mcds"

    def __init__(self,  name="mcds_Bikit", cache_dir=None, split_type="", transform=None,
                 load_all_in_mem=False, devel_mode=False):
        """

        :param name: Dataset name.
        :param split_type: Use 'trainval_bikit' or 'test_bikit', or '' (default) for all samples. Disclaimer: There are
                no original splits. The authors of bikit introduced this splits for comparability. Due to the dataset
                size it is recommended to do cross-validation on the trainvalid split.
        :param transform: Torch transformation for image data (this depends on your CNN).
        :param cache_dir: Path to cache_dir.
        :param load_all_in_mem: Whether or not to load all image data into memory (this depends on the dataset size and
            your memory). Loading all in memory can speed up your training.
        :param devel_mode: Only using 100 samples.
        """
        assert name in list(self.DATASETS.keys()), f"This name does not exists. Use something from {list(DATASETS.keys())}."
        self.name = name
        assert split_type in ["", "trainval_bikit", "test_bikit"], f'You used split_type str({split_type}). Only ["", "trainval_bikit", "test_bikit"] are allowed.'
        bikit_path = dirname(dirname(__file__))
        self.csv_filename = os.path.join(bikit_path, "data", self.name) + ".csv"

        # Misc
        self.split_type = split_type
        if cache_dir:
            self.cache_full_dir = os.path.join(cache_dir)
        else:
            self.cache_full_dir = os.path.join(os.path.expanduser("~"), ".bikit")

        self.devel_mode = devel_mode
        self.class_names = ['Cracks', 'Efflorescence', 'Scaling', 'Spalling', 'General', 'NoDefect',
                            'ExposedReinforcement', 'NoExposedReinforcement', 'RustStaining', 'NoRustStaining']
        self.num_classes = 10
        self.load_all_in_mem = load_all_in_mem

        # Data prep
        self.transform = transform

        self.df = pd.read_csv(self.csv_filename)
        if split_type:
            self.df = self.df[self.df["split_type"] == split_type]
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
        img_filename = os.path.join(self.cache_full_dir, data['img_path'])
        img = pil_loader(img_filename)
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        # Get label with shape 10
        label = torch.FloatTensor(data[self.class_names].to_numpy().astype("float32"))
        return img, label

    def __len__(self):
        return self.n_samples

if __name__ == "__main__":
    print(__file__)
    all_dataset = McdsDataset(split_type="")
    trainval_dataset = McdsDataset(split_type="trainval_bikit")
    test_dataset = McdsDataset(split_type="test_bikit")
    development_dataset = McdsDataset(split_type="test_bikit", devel_mode=True)

    print("===")
    img, targets = all_dataset[0]
    print(len(all_dataset))
    print(img.shape, targets.shape)
    print(list(targets.shape))
    print(targets.dtype)
    print("===")

    print(len(all_dataset) )
    print(len(trainval_dataset) )
    print(len(test_dataset))
    print(len(development_dataset))
    print("===Done===")

    all_dataset = McdsDataset(name="mcds_Bukhsh", split_type="")
    trainval_dataset = McdsDataset(name="mcds_Bukhsh", split_type="trainval_bikit")
    test_dataset = McdsDataset(name="mcds_Bukhsh", split_type="test_bikit")
    development_dataset = McdsDataset(name="mcds_Bukhsh", split_type="test_bikit", devel_mode=True)

    print(len(all_dataset) )
    print(len(trainval_dataset) )
    print(len(test_dataset))
    print(len(development_dataset))