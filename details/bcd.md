# BCD

This data set was used in the articel "Automated Bridge Crack Detection Using Convolutional Neural Networks"
by [Xu H, Su X, Wang Y, et al.](https://www.mdpi.com/2076-3417/9/14/2867)

The original bridge crack data set was artificially enhanced to generate the data set used in the paper by
[Li Liang-Fu, Ma Wei-Fei, Li Li, Lu Cheng. Research on detection algorithm for bridge cracks based on deep learning.
Acta Automatica Sinica, 2018]

The original dataset contains a total of 4,856 training pictures and 1213 test pictures are included,
with a resolution of 224*224.

## bcd in bikit

Bikit introduces his own split of the bridge crack detection dataset.

        
|              | Train | Val | Test |
|--------------|:------|:----|:-----|
|**Cracks**    | 1414  | 300 |  300 |
|**No Cracks** | 3455  | 300 |  300 |