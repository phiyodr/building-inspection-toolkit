# Bridge Inspection Toolkit

| Testing | 
| :-----: | 
| [![Build Status](https://travis-ci.com/phiyodr/bridge-inspection-toolkit.svg?branch=master)](https://travis-ci.com/phiyodr/bridge-inspection-toolkit) | 


**Bridge Inspection Toolkit** helps you with dataset handling in the field for Damage Detection for Reinforced Concrete Bridges.
This DataHub is build for [PyTorch](https://pytorch.org/). 

# The Datasets


| Name | Type |  Size | Implemented
|----|-----|---|---|
CDS  | Binary Clf  | 1,027 | not yet
SDNETv1  | Binary Clf  | 13,620 | not yet
BCD  | Binary Clf  | 5,390 | not yet
ICCD  | Binary Clf  | 60,010 | not yet
MCDS  | 10-Class Clf  | 3,617 | **yes**
[CODEBRIM](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html)  | 6-Class Multi-target Clf  | 7261 | **yes**
COCOBridge  | 4-Class OD  | 774/+2,500 | not yet


# Use the Application

```python
from bikit.utils import list_datasets, download_dataset

# List all datasets
list_datasets()

# Download data
download_dataset("<name>") 
```

* `mcds`


```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import McdsDataset
from torch.utils.data import DataLoader
from torchvision import transforms

download_dataset("mcds") 
my_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
trainval_dataset = McdsDataset(split_type="trainval", transform=my_transform)
trainval_loader = DataLoader(dataset=trainval_dataset, batch_size=64, shuffle=False, num_workers=0)

# Use it in your training loop
for i, (imgs, labels) in enumerate(trainval_loader):
	print(i, imgs.shape, labels.shape)
```

* `codebrim-classif-balanced`

```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import CodebrimDataset
download_dataset("codebrim-classif-balanced") # Takes quite a time
train_dataset = CodebrimDataset(split_type="train")
```

# PyTest

Run PyTest

```bash
# cd bridge-inspection-toolkit/
pytest
```



#### Repo

The repo stucture is based on https://github.com/sisl/python_package_template.
