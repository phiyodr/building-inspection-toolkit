# Bridge Inspection Toolkit


[![build](https://travis-ci.com/phiyodr/bridge-inspection-toolkit.svg?branch=master)](https://travis-ci.com/phiyodr/bridge-inspection-toolkit) 
[![GitHub license](https://img.shields.io/github/license/phiyodr/bridge-inspection-toolkit.svg)](https://github.com/phiyodr/bridge-inspection-toolkit/blob/master/LICENSE) 
[![GitHub tag](https://img.shields.io/github/tag/phiyodr/bridge-inspection-toolkit.svg)](https://GitHub.com/phiyodr/bridge-inspection-toolkit/tags/)



**Bridge Inspection Toolkit** helps you with dataset handling in the field for Damage Detection for Reinforced Concrete Bridges.
This DataHub is build for [PyTorch](https://pytorch.org/). 

# The Datasets


## Publicly available datasets

Name      | Type        | Unique images | Implemented | Fix eval set
----------|-------------|---------------|-------------|-------------
CDS       | Binary Clf  |            1k |     not yet | ukn
SDNETv1   | Binary Clf  |           13k |     not yet | ukn
BCD  [[Paper]](https://www.mdpi.com/2076-3417/9/14/2867)  [[Data]](https://github.com/tjdxxhy/crack-detection)   | Binary Clf  |            5k |     not yet | yes
ICCD      | Binary Clf  |           60k |     not yet | ukn
MCDS [[Paper]](https://www.researchgate.net/publication/332571358_Multi-classifier_for_Reinforced_Concrete_Bridge_Defects) [[Data]](https://zenodo.org/record/2601506)  | 10-Class Clf  | 3,617 | **yes** | no
CODEBRIM [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html) [[Data]](https://zenodo.org/record/2620293#.YO8rj3UzZH4) | 6-Class Multi-target Clf  | 7261 | **yes** | yes
COCOBridge | 4-Class OD | 774/+2,500    |     not yet | ukn

## Different dataset versions (`name`) and different splits (`split`)

**Different versions**

For some datasets different versions exists. This may be due to the fact that the authors already provide different version (e.g. CODEBRIM) or other authors update datasets (e.g. Bukhsh for MCDS). Moreover we introduce 

**Splits** 

We provide carefully selected *train/valid/test* (for large datasets) resp. *trainval/test splits* (for small datasets) to create comparability for these datasets. That means that we introduce splits, when they are not available or update spits where we think this is useful. 

**Overview**


| `name`                      | `split`                               | Note |
| ----------------------------|---------------------------|-------------------------------|
| `mcds_Bukhsh`               | No original splits available. | Bukhsh et al. creates a 10 class dataset out of the 3-step dataset from Hüthwohl et al.  |
|                             | **`bikit`** with `trainval` and `test` | |
| `mcds_Bikit`                | **`bikit`** with `trainval` and `test` |
| `codebrim-classif-balanced` | **`original`** with `train`, `valid`, `test` | Underrepresented classes are oversampled.  |
|                             | **`bikit`** with `train`, `valid`, `test` | **TODO** Not implemented yet |

<!-- `codebrim-classif`          |  **`original`** with `train`, `valid`, `test` | Plain original version | -->

* For large datasets we use `train` for training, `valid` for validation and `test` for testing.
* For small datasets we use `trainval` for training and validation (you have to do [CV](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) on your own) and `test` for testing.


# Use the Application

```python
from bikit.utils import list_datasets, download_dataset

# List all datasets
list_datasets()

# Download data
download_dataset("<name>") 
```

### `mcds_Bukhsh`

The original version from Hüthwohl‬ et al. is a sequential 3-step approach, which is not provided. [Bukhsh et al.](https://link.springer.com/article/10.1007/s00521-021-06279-x) structure it as a 10-class problem.


[More details](/details/mcds.md)

```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import McdsDataset
from torch.utils.data import DataLoader
from torchvision import transforms

download_dataset("mcds_Bukhsh") # equal to `download_dataset("mcds_Bikit")` 
my_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
trainval_dataset = McdsDataset(split="trainval", transform=my_transform)
trainval_loader = DataLoader(dataset=trainval_dataset, batch_size=64, shuffle=False, num_workers=0)

# Use it in your training loop
for i, (imgs, labels) in enumerate(trainval_loader):
	print(i, imgs.shape, labels.shape)
```

### `mcds_Bikit`

A cleaned version of `mcds_Bukhsh`.  

[More details](/details/mcds.md).

```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import McdsDataset
from torch.utils.data import DataLoader
from torchvision import transforms

download_dataset("mcds_Bikit") # equal to `download_dataset("mcds_Bukhsh")` 
my_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
trainval_dataset = McdsDataset(split="trainval", transform=my_transform)
trainval_loader = DataLoader(dataset=trainval_dataset, batch_size=64, shuffle=False, num_workers=0)

# Use it in your training loop
for i, (imgs, labels) in enumerate(trainval_loader):
	print(i, imgs.shape, labels.shape)
```


### `codebrim-classif-balanced`

Original version from [Mundt et al](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html).

```python
from bikit.utils import download_dataset
from bikit.datasets.mcds import CodebrimDataset
download_dataset("codebrim-classif-balanced") # Takes quite a time
train_dataset = CodebrimDataset(split="train")
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

The repo structure is based on https://github.com/sisl/python_package_template.
