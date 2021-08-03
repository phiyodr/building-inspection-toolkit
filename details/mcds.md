# MCDS

This dataset comes from [Hüthwohl et al](https://www.researchgate.net/publication/332571358_Multi-classifier_for_Reinforced_Concrete_Bridge_Defects).

It is a Multi-class, multi-label classification dataset.


| 1st Stage       | 2nd Stage             | 3rd Stage     |
|-----------------|-----------------------|---------------|
| Crack           |                       |               |
| Efflorescence   | ⟶ ExposedReinforcement/NoExposedReinforcement | ⟶ RustStaining/NoRustStaining |
| Scaling         | ⟶ ExposedReinforcement/NoExposedReinforcement | ⟶ RustStaining/NoRustStaining |
| Spalling        | ⟶ ExposedReinforcement/NoExposedReinforcement | ⟶ RustStaining/NoRustStaining |
| General defect  |                       |               |
| No defect       |                       |               |


## `mcds_Bukhsh`

Version from [Bukhsh1 et al](https://link.springer.com/content/pdf/10.1007/s00521-021-06279-x.pdf).




Combinations of labels, total appearance of labels in bottom line:


| Cra.   | Eff. | Sca. | Spa. | Gen.| NoDef. | ExpRe. | NoExpRe. | RustS. | NoRustS. | COUNTS | trainval (bikit) |   test (bikit) |
|--------|:-----|:-----|:-----|:----|:-------|:-------|:---------|:-------|:---------|:-------|:-----------------|:--------------|
| x      |      | x    |      |     |        |        |          |        |          | 1      |                0 |             1 |
| x      |      |      |      |     |        |        |          | x      |          | 1      |                0 |             1 |
| x      |      |      |      |     |        |        |          |        | x        | 132    |               82 |            50 |
| x      |      |      |      |     |        |        |          |        |          | 655    |              605 |            50 |
|        | x    | x    |      |     |        |        |          |        | x        | 1      |                0 |             1 |
|        | x    | x    |      |     |        |        |          |        |          | 3      |                0 |             1 |
|        | x    |      | x    |     |        | x      |          | x      |          | 2      |                0 |             1 |
|        | x    |      | x    |     |        |        | x        |        |          | 1      |                0 |             1 |
|        | x    |      |      |     |        |        |          |        | x        | 57     |               51 |             6 |
|        | x    |      |      |     |        |        |          |        |          | 247    |              197 |            50 |
|        |      | x    |      |     |        |        |          |        | x        | 29     |               26 |             3 |
|        |      | x    |      |     |        |        |          |        |          | 134    |               84 |            50 |
|        |      |      | x    |     |        | x      |          | x      |          | 145    |               95 |            50 |
|        |      |      | x    |     |        | x      |          |        | x        | 51     |               45 |             6 |
|        |      |      | x    |     |        | x      |          |        |          | 25     |               22 |             3 |
|        |      |      | x    |     |        |        | x        | x      |          | 1      |                0 |             1 |
|        |      |      | x    |     |        |        | x        |        |          | 201    |              151 |            50 |
|        |      |      | x    |     |        |        |          | x      |          | 1      |                0 |             1 |
|        |      |      |      | x   |        |        |          |        | x        | 47     |               42 |             5 |
|        |      |      |      | x   |        |        |          |        |          | 217    |              167 |            50 |
|        |      |      |      |     | x      |        |          |        | x        | 94     |               84 |            10 |
|        |      |      |      |     | x      |        |          |        |          | 358    |              308 |            50 |
|        |      |      |      |     |        |        |          | x      |          | 205    |              155 |            50 |
|        |      |      |      |     |        |        |          |        | x        | 4      |                0 |             1 |
| **789**	| **311**	| **168**	| **427**	| **264**	| **452**	| **223**	| **203**	| **355**	| **415**	| total=*3607* | total=*2114* | total=*498*



## `mcds_Bikit` 

Combinations of labels, total appearance of labels in bottom line:


| Cra. | Eff. | Sca. | Spa. | Gen. | NoDef. | ExpRe. | RustS. | COUNTS | trainval (bikit) |   test (bikit) |
|:-----|:-----|:-----|:-----|:-----|:-------|:-------|:-------|-------:|---------:|----:|
| x    |      |      |      |      |        |        |        | 787    | 737 | 50 |
|      | x    |      |      |      |        |        |        | 304    | 254 | 50 |
|      |      | x    |      |      |        |        |        | 163    | 113 | 50 |
|      |      |      | x    |      |        | x      | x      | 145    |  95 | 50 |
|      |      |      | x    |      |        | x      |        |  76    |  26 | 50 |
|      |      |      | x    |      |        |        |        | 201    | 151 | 50 |
|      |      |      |      | x    |        |        |        | 264    | 214 | 50 |
|      |      |      |      |      | x      |        |        | 452    | 402 | 50 |
|      |      |      |      |      |        |        | x      | 205    | 155 | 50 |
| **787**  | **304**  | **163**  | **422**  | **264**  | **452**    | **221**    | **350**    | total=*2597* | total=*2147* | total=*450*

This version has two major differences to `mcds_Bukhsh`: 

* First of all, the classes `NoExposedReinforcement` and `NoRustStaining` were removed, since only a few images without Exposed Reinforcement or without Rust also had the corresposing label. It is possible to impute the missing values (i.e. for each image either `ExposedReinforcement` or  `NoExposedReinforcement` must be present) but we decided not to have "negative labels" in our dataset version (it is not common to have negative labels for a subset of labels). 
* Then we removed combinations with frequencies lower 5. For example there was one image with label `Crack` and `Scaling`. Moreover we removed remaining images with no labels, i.e. images which only had the labels `NoExposedReinforcement` and/or `NoRustStaining`.  

Notes: Contrary to what is mentioned in the Hüthwohl et al., there are images that only have the label `RustStaining` (3rd stage) without having positive labels from stage 1 or 2. After a visual inspection we decided to include these images since the label is correct. 
