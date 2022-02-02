# Bridge Inspection Toolkit


[![build](https://travis-ci.com/phiyodr/bridge-inspection-toolkit.svg?branch=master)](https://travis-ci.com/phiyodr/bridge-inspection-toolkit) 
[![codecov](https://codecov.io/gh/phiyodr/bridge-inspection-toolkit/branch/master/graph/badge.svg?token=U685JTKNLC)](https://codecov.io/gh/phiyodr/bridge-inspection-toolkit)
[![GitHub license](https://img.shields.io/github/license/phiyodr/bridge-inspection-toolkit.svg)](https://github.com/phiyodr/bridge-inspection-toolkit/blob/master/LICENSE) 
[![GitHub tag](https://img.shields.io/github/tag/phiyodr/bridge-inspection-toolkit.svg)](https://GitHub.com/phiyodr/bridge-inspection-toolkit/tags/)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)



**Bridge Inspection Toolkit** helps you with dataset handling in the field for Damage Detection for Reinforced Concrete Bridges.
This DataHub is build for [PyTorch](https://pytorch.org/). 

# The Datasets


## Publicly available datasets

Name      | Type        | Unique images | Fix eval set
----------|-------------|---------------|-------------
CDS   [[Web]](https://www.repository.cam.ac.uk/handle/1810/267902)    | Binary Clf  |            1k | ukn
SDNET  [[Web]](https://digitalcommons.usu.edu/all_datasets/48/)    | Binary Clf  |           56k | bik
BCD  [[Paper]](https://www.mdpi.com/2076-3417/9/14/2867)  [[Data]](https://github.com/tjdxxhy/crack-detection)   | Binary Clf  |            6k | yes
MCDS [[Paper]](https://www.researchgate.net/publication/332571358_Multi-classifier_for_Reinforced_Concrete_Bridge_Defects) [[Data]](https://zenodo.org/record/2601506)  | 10-Class Clf  | 3,617 | no, bikit
CODEBRIM [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html) [[Data]](https://zenodo.org/record/2620293#.YO8rj3UzZH4) | 6-Class Multi-target Clf  | 7261 | **yes** | yes

Missing:  ICCD (Binary Clf, 60k), COCOBridge (4-Class OD, 774/+2,500)

## Different dataset versions (`name`) and different splits (`split`)

**Different versions**

For some datasets different versions exists. This may be due to the fact that the authors already provide different version (e.g. CODEBRIM) or other authors update datasets (e.g. Bukhsh for MCDS). 

**Splits** 

We provide carefully selected *train/valid/test* (for large datasets). We introduce splits, when they are not available or update spits where we think this is useful. 

**Overview**


| `name`                      | `split`                               | Note |
| ----------------------------|----------------------------|-------------------------------|
| `cds`                       | `["train", "val", "test"]` |          
| `bcd`                       | `["train", "val", "test"]` |     
| `sdnet`                     | `["train", "val", "test"]` | Many wrong labels        
| `sdnet_binary`              | `["train", "val", "test"]` | Many wrong labels; Binaried version of sdnet: crack, no crack
| `sdnet_bikit`               | `["train", "val", "test"]` | Cleaned bikit version     
| `sdnet_bikit_binary`        | `["train", "val", "test"]` | Cleaned bikit version; Binaried version of sdnet: crack, no crack             
| `mcds_Bukhsh`               | `["trainval", "test"]`     | Bukhsh et al. creates a 10 class dataset out of the 3-step dataset from Hüthwohl et al.  |
| `mcds_bikit`                | `["train", "val", "test"]` | Cleaned bikit version
| `codebrim-classif-balanced` | `["train", "val", "test"]` | Underrepresented classes are oversampled.  |
| `codebrim-classif`          | `["train", "val", "test"]` | Original set  |
| `meta3`		        	  | `["train", "val", "test"]` | 6-class multi-target dataset based on bcd, codebrim-classif, and mcds_bikit. |
| `meta4`       		   	  | `["train", "val", "test"]` | 6-class multi-target dataset based on bcd, codebrim-classif, mcds_bikit, and sdnet_bikit_binary.  |


# Use the Application

```python
from bikit.utils import list_datasets, download_dataset

# List all datasets
list_datasets()

# Download data
download_dataset("<name>") 
```

### Demo

```python
from bikit.utils import download_dataset
from bikit.datasets.data import BikitDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Select a dataset:
all_datasets =list_datasets()
name = "mcds_bikit"

download_dataset(name) # equal to `download_dataset("mcds_Bukhsh")` 
my_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
# Use return_type 'pt' (default) or 'np'
train_dataset = BikitDataset(name, split="train", transform=my_transform, return_type="pt") 
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)

# Use it in your training loop
for i, (imgs, labels) in enumerate(train_dataset):
	print(i, imgs.shape, labels.shape)
```

# Misc

### PyTest

Install dependencies first

```bash
pip3 install -U -r requirements.txt -r test_requirements.txt
```

Run PyTest

```bash
# cd bridge-inspection-toolkit/
pytest
```