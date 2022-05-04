# Building Inspection Toolkit


[![build](https://travis-ci.com/phiyodr/bridge-inspection-toolkit.svg?branch=master)](https://travis-ci.com/phiyodr/bridge-inspection-toolkit) 
[![codecov](https://codecov.io/gh/phiyodr/bridge-inspection-toolkit/branch/master/graph/badge.svg?token=U685JTKNLC)](https://codecov.io/gh/phiyodr/bridge-inspection-toolkit)
[![GitHub license](https://img.shields.io/github/license/phiyodr/bridge-inspection-toolkit.svg)](https://github.com/phiyodr/bridge-inspection-toolkit/blob/master/LICENSE) 
[![GitHub tag](https://img.shields.io/github/tag/phiyodr/bridge-inspection-toolkit.svg)](https://GitHub.com/phiyodr/bridge-inspection-toolkit/tags/)
[![Project Status: WIP](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Twitter](https://img.shields.io/twitter/url?url=https%3A%2F%2Fshields.io)](https://twitter.com/dacl_ai)

**Building Inspection Toolkit** helps you with machine learning projects in the field of damage recognition on built structures, currenlty with a focus on bridges.


* [DataHub](#data): It contains currated open-source datasets, with fix train/val/test splits (as this is often missing) and cleaned annotations. It is build for [PyTorch](https://pytorch.org/). 
* [Metrics](#metrics): We define useful metrics you can use and report to make comparability easier.
* [Pre-trained Models](#models): We provide strong baseline models for different datasets. See [bikit-models](https://github.com/phiyodr/bikit-models) for more details.


# The Datasets


## Open-source data

Name      | Type        | Unique images | train/val/test split
----------|-------------|---------------|-------------
CDS   [[Web]](https://www.repository.cam.ac.uk/handle/1810/267902)    | 2-class single-target Clf  |            1,028 | bikit-version
BCD  [[Paper]](https://www.mdpi.com/2076-3417/9/14/2867)  [[Data]](https://github.com/tjdxxhy/crack-detection)   | 2-class single-target Clf  |            6069 | modified-version
SDNET  [[Web]](https://digitalcommons.usu.edu/all_datasets/48/)    | 2-class single-target Clf  |           56,092 | bikit-version
MCDS [[Paper]](https://www.researchgate.net/publication/332571358_Multi-classifier_for_Reinforced_Concrete_Bridge_Defects) [[Data]](https://zenodo.org/record/2601506)  | 8-class multi-target Clf  | 2,612 | bikit-version
CODEBRIM [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/Mundt_Meta-Learning_Convolutional_Neural_Architectures_for_Multi-Target_Concrete_Defect_Classification_With_CVPR_2019_paper.html) [[Data]](https://zenodo.org/record/2620293#.YO8rj3UzZH4) | 6-class multi-target Clf  | 7,730 | original-version

<!--Missing:  ICCD (Binary Clf, 60k), COCOBridge (4-Class OD, 774/+2,500)-->

## Bikit datasets

**Different versions**

For some datasets different versions exists. This may be due to the fact that the authors already provide different version (e.g. CODEBRIM) or other authors update datasets (e.g. Bukhsh for MCDS). 

**Splits** 

We provide carefully selected *train/val/test* splits. We introduce splits, when they are not available or update splits where we think this is useful. 

**Overview**


| `name`                      | Note 																								  |
| ----------------------------|-------------------------------------------------------------------------------------------------------|
| `cds`                       | *Original dataset* with bikit's *train/val/test* splits         
| `bcd`                       | *Original dataset* with modified *train/val/test* splits (original train was splitted into *train/val*)      
| `sdnet`                     | *Original dataset* with bikit's *train/val/test* splits; Many wrong labels    
| `sdnet_binary`              |  Many wrong labels; Binarized version of sdnet: crack, no crack
| `sdnet_bikit`               |  Cleaned wrong labels     
| `sdnet_bikit_binary`        |  Cleaned wrong labels; Binarized version of sdnet: crack, no crack             
| `mcds_Bukhsh`               |  Bukhsh et al. create a 10-class dataset out of the 3-step dataset from Hüthwohl et al. (with same wrong labels); With bikit's *train/val/test* splits  |
| `mcds_bikit`                |  We create a 8-class dataset from Hüthwohl et al. which prevent wrong labels with bikit's *train/val/test* splits. 
| `codebrim-classif-balanced` | *Original dataset* with original *train/val/test* splits: Underrepresented classes are oversampled.  |
| `codebrim-classif`          | *Original dataset* with original *train/val/test* splits. |
| `meta3`		        	  |  6-class multi-target dataset based on bcd, codebrim-classif, and mcds_bikit. |
| `meta4`       		   	  |  6-class multi-target dataset based on bcd, codebrim-classif, mcds_bikit, and sdnet_bikit_binary.  |


# Usage

## Data

**List and download data**

```python
from bikit.utils import list_datasets, download_dataset

# List all datasets
list_datasets()

# Download data
download_dataset("<name>") 
```

**Use `BikitDataset`**

```python
from bikit.utils import download_dataset
from bikit.datasets import BikitDataset # Deprecated: from bikit.datasets.data import BikitDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Select a dataset:
name = "mcds_bikit"

download_dataset(name) # equals to `download_dataset("mcds_Bukhsh")` 
my_transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
# Use return_type 'pt' (default) or 'np'
train_dataset = BikitDataset(name, split="train", transform=my_transform, return_type="pt") 
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, num_workers=0)

# Use it in your training loop
for i, (imgs, labels) in enumerate(train_dataset):
	print(i, imgs.shape, labels.shape)
	break
```

## Metrics

* For single-target problems (like `cds`, `bcd`, `sdnet_bikit_binary`) metrics will follow (#TODO).
* For multi-target problems (like `sdnet_bikit`, `mcds_bikit` or `meta3`) we use **Exact Match Ratio** (`EMR_mt`) and **classwise Recall** (`Recalls_mt`).


```python
#!pip install torchmetrics
from bikit.metrics import EMR_mt, Recalls_mt
myemr = EMR_mt(use_logits=False)
myrecalls = Recalls_mt()

# fake data
preds0  = torch.tensor([[.9, 0.1, 0.9, 0.1, 0.9, 0.1], 
                       [.8, 0.2, 0.9, 0.2, 0.9, 0.2], 
                       [.7, 0.9, 0.2 , 0.2, 0.2 , 0.2]])
preds1 = torch.tensor([[.0, 0.1, 0.9, 0.1, 0.9, 0.1], 
                       [.8, 0.2, 0.9, 0.2, 0.9, 0.2], 
                       [.7, 0.9, 0.2 , 0.9, 0.2 , 0.9]])
target = torch.tensor([[1, 0, 1, 0, 0, 1], 
                      [1, 1, 0, 0, 1, 0], 
                      [1, 1, 0, 1, 0, 1]])
# batch 0
myemr(preds0, target), myrecalls(preds0, target)
print(myemr.compute(), myrecalls.compute())

# batch 1
myemr(preds1, target), myrecalls(preds1, target)    
print(myemr.compute(), myrecalls.compute())

# Reset at end of epoch
myemr.reset(), myrecalls.reset()
print(myemr, myrecalls)
```


## Models

**List models**

```python
from bikit.utils import list_models

# List all models
list_models()

# Download and load model
model, metadata = load_model("MCDS_ResNet50")
```

**Model Inference**

```python
from bikit.utils import load_model, get_metadata, load_img_from_url
from bikit.models import make_prediction

img = load_img_from_url("https://github.com/phiyodr/building-inspection-toolkit/raw/master/bikit/data/11_001990.jpg")
model, metadata = load_model("MCDS_ResNet50", add_metadata=True)
prob, pred = make_prediction(model, img, metadata, print_predictions=True, preprocess_image=True)
#> Crack                [██████████████████████████████████████  ]  95.86% 
#> Efflorescence        [                                        ]   0.56% 
#> ExposedReinforcement [                                        ]   0.18% 
#> General              [                                        ]   0.60% 
#> NoDefect             [                                        ]   1.29% 
#> RustStaining         [                                        ]   0.44% 
#> Scaling              [                                        ]   0.05% 
#> Spalling             [                                        ]   0.85% 
#> Inference time (CPU): 44.26 ms
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

### Citation

Use the "Cite this repository" tool in the *About* section of this repo to cite us :)
