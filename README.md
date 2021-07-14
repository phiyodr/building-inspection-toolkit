# Bridge Inspection Toolkit

| Testing | 
| :-----: | 
| [![Build Status](https://travis-ci.com/phiyodr/bridge-inspection-toolkit.svg?branch=master)](https://travis-ci.com/phiyodr/bridge-inspection-toolkit) | 


**Bridge Inspection Toolkit** helps you with dataset handling in the field for Damage Detection for Reinforced Concrete Bridges.
This DataHub is build for [PyTorch](https://pytorch.org/). 

# The Datasets


## Publicly available datasets

| Name | Type |  Unique images | Implemented | Fix eval set
|----|-----|---|---|
CDS  | Binary Clf  | 1k | not yet | ukn
SDNETv1  | Binary Clf  | 13k | not yet | ukn
BCD  | Binary Clf  | 5k | not yet | ukn
ICCD  | Binary Clf  | 60k | not yet | ukn
MCDS [Paper](https://www.researchgate.net/publication/332571358_Multi-classifier_for_Reinforced_Concrete_Bridge_Defects) [Data](https://zenodo.org/record/2601506)  | 10-Class Clf  | 3,617 | **yes** | no
CODEBRIM [Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html) [Data](https://zenodo.org/record/2620293#.YO8rj3UzZH4) | 6-Class Multi-target Clf  | 7261 | **yes** | yes
COCOBridge  | 4-Class OD  | 774/+2,500 | not yet | ukn


## Different dataset versions (`name`) and differnt splits (`split_style`)

**Splits** 

We want to provide carefully selected train/valid/test resp. trainval/test splits (where dataset size is small) to create comparability for theses datasets. That means that we introduce splits where they are not available or update spits where we think this is useful. 


**Different versions**

For some datasets differnt versions exists. This may be that the authors already provide different version (e.g. CODEBRIM) or other authors update datasets (e.g. Bukhsh for MCDS).


| `name` | `split_type`s with splits | Note |
|--------|---------------------------|-------------------------------|
`mcds_Bukhsh` | No original splits available. **`bikit`** with `trainval` (for [CV](https://en.wikipedia.org/wiki/Cross-validation_(statistics))) and `test`  | Bukhsh et al. creates a 10 class dataset out of the 3-step dataset from Hüthwohl et al.  |
`mcds_bikit` | **`bikit`** with `trainval` (for [CV](https://en.wikipedia.org/wiki/Cross-validation_(statistics))) and `test` |
`codebrim-classif-balanced` | **`original`** with `train`, `valid`, `test` | Underrepresented classes are oversampled.  |
`codebrim-classif` |  **`original`** with `train`, `valid`, `test` | Underrepresented classes are oversampled. (**TODO**) |



# Use the Application

```python
from bikit.utils import list_datasets, download_dataset

# List all datasets
list_datasets()

# Download data
download_dataset("<name>") 
```

### `mcds_Bukhsh`

The orginal version is a sequential 3-step approach, which is not provided. Bukhsh et al. structure it as a 10-class problem.

```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import McdsDataset
from torch.utils.data import DataLoader
from torchvision import transforms

download_dataset("mcds_Bukhsh") # equal to `download_dataset("mcds_Bikit")` 
my_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
trainval_dataset = McdsDataset(split_type="trainval", transform=my_transform)
trainval_loader = DataLoader(dataset=trainval_dataset, batch_size=64, shuffle=False, num_workers=0)

# Use it in your training loop
for i, (imgs, labels) in enumerate(trainval_loader):
	print(i, imgs.shape, labels.shape)
```

### `mcds_Bikit`


```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import McdsDataset
from torch.utils.data import DataLoader
from torchvision import transforms

download_dataset("mcds_Bikit") # equal to `download_dataset("mcds_Bukhsh")` 
my_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
trainval_dataset = McdsDataset(split_type="trainval", transform=my_transform)
trainval_loader = DataLoader(dataset=trainval_dataset, batch_size=64, shuffle=False, num_workers=0)

# Use it in your training loop
for i, (imgs, labels) in enumerate(trainval_loader):
	print(i, imgs.shape, labels.shape)
```


### `codebrim-classif-balanced`

Original version from Hütwowhl et al.

```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import CodebrimDataset
download_dataset("codebrim-classif-balanced") # Takes quite a time
train_dataset = CodebrimDataset(split_type="train")
```

# PyTest

Install dependencies

```bash
pip3 install -U -r requirements.txt -r test_requirements.txt
```

Run PyTest

```bash
# cd bridge-inspection-toolkit/
pytest
```



#### Repo

The repo stucture is based on https://github.com/sisl/python_package_template.
