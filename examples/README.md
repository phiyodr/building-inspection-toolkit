# Click the badges below to access the notebooks

| multi-target    | Link |
|-----------------|------|
| Inference       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/phiyodr/building-inspection-toolkit/blob/master/examples/colab/multi_target.ipynb) |


## Available Models

The currently available models are displayed in the table below. They are sorted according to the Exact Match Ratio (EMR), which is the most important metric for multi-target classification. Further information reagarding the models and the metrics may be found on [dacl.ai](https://dacl.ai/bikit.html) and in the [*bikit*-paper](https://arxiv.org/abs/2202.07012).

| Modelname                             | Dataset                   | EMR   | F1   | Tag          |CorrespNameOnDaclAI*  |
|---------------------------------------|---------------------------|-------|------|--------------|----------------------|
| CODEBRIMbalanced_ResNet50_hta         | codebrim-classif-balanced | 73.73 | 0.85 | ResNet       |Code_res_dacl         |
| CODEBRIMbalanced_MobileNetV2          | codebrim-classif-balanced |70.41  | 0.84 | MobileNetV2  |Code_mobilev2_dacl    |
| CODEBRIMbalanced_MobileNetV3Large_hta | codebrim-classif-balanced | 69.46 | 0.83 | MobileNet    |Code_mobile_dacl      |
| CODEBRIMbalanced_EfficientNetV1B0_dhb | codebrim-classif-balanced | 68.67 | 0.84 | EfficientNet |Code_eff_dacl         |
| MCDSbikit_MobileNetV3Large_hta        | mcds_bikit                | 54.44 | 0.66 | MobileNet    |McdsBikit_mobile_dacl |
| MCDSbikit_EfficientNetV1B0_dhb        | mcds_bikit                | 51.85 | 0.65 | EfficientNet |McdsBikit_eff_dacl    |
| MCDSbikit_ResNet50_dhb                | mcds_bikit                | 48.15 | 0.62 | ResNet       |McdsBikit_res_dacl    |

**CorrespNameOnDaclAI* displays the shortened name under which you can find the model on [*dacl.ai*](https://dacl.ai).