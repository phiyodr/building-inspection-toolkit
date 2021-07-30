# MCDS

Multi-class, multi-label classification 


| 1st Stage       | 2nd Stage             | 3rd Stage     |
|-----------------|-----------------------|---------------|
| Crack           |                       |               |
| Efflorescence   | ⟶ ExposedReinforcement/NoExposedReinforcement | ⟶ RustStaining/NoRustStaining |
| Scaling         | ⟶ ExposedReinforcement/NoExposedReinforcement | ⟶ RustStaining/NoRustStaining |
| Spalling        | ⟶ ExposedReinforcement/NoExposedReinforcement | ⟶ RustStaining/NoRustStaining |
| General defect  |                       |               |
| No defect       |                       |               |


## `mcds_Bukhsh`


Combinations of labels, total appearance of labels in bottom line:


| Cra.   | Eff.   | Sca.   | Spa.   | Gen.   | NoDef.   | ExpRe.   | NoExpRe.   | RustS.   | NoRustS.   | COUNTS   |
|--------|:----------------|:----------|:-----------|:----------|:-----------|:-----------------------|:-------------------------|:---------------|:-----------------|:----|
| x        |                 | x         |            |           |            |                        |                          |                |                  | 1   |
| x        |                 |           |            |           |            |                        |                          | x              |                  | 1   |
| x        |                 |           |            |           |            |                        |                          |                | x                | 132 |
| x        |                 |           |            |           |            |                        |                          |                |                  | 655 |
|          | x               | x         |            |           |            |                        |                          |                | x                | 1   |
|          | x               | x         |            |           |            |                        |                          |                |                  | 3   |
|          | x               |           | x          |           |            | x                      |                          | x              |                  | 2   |
|          | x               |           | x          |           |            |                        | x                        |                |                  | 1   |
|          | x               |           |            |           |            |                        |                          |                | x                | 57  |
|          | x               |           |            |           |            |                        |                          |                |                  | 247 |
|          |                 | x         |            |           |            |                        |                          |                | x                | 29  |
|          |                 | x         |            |           |            |                        |                          |                |                  | 134 |
|          |                 |           | x          |           |            | x                      |                          | x              |                  | 145 |
|          |                 |           | x          |           |            | x                      |                          |                | x                | 51  |
|          |                 |           | x          |           |            | x                      |                          |                |                  | 25  |
|          |                 |           | x          |           |            |                        | x                        | x              |                  | 1   |
|          |                 |           | x          |           |            |                        | x                        |                |                  | 201 |
|          |                 |           | x          |           |            |                        |                          | x              |                  | 1   |
|          |                 |           |            | x         |            |                        |                          |                | x                | 47  |
|          |                 |           |            | x         |            |                        |                          |                |                  | 217 |
|          |                 |           |            |           | x          |                        |                          |                | x                | 94  |
|          |                 |           |            |           | x          |                        |                          |                |                  | 358 |
|          |                 |           |            |           |            |                        |                          | x              |                  | 205 |
|          |                 |           |            |           |            |                        |                          |                | x                | 4   |
| **789**	| **311**	| **168**	| **427**	| **264**	| **452**	| **223**	| **203**	| **355**	| **415**	| total=*3607*

## `mcds_Bikit` 

Combinations of labels, total appearance of labels in bottom line:


| Cra. | Eff. | Sca. | Spa. | Gen. | NoDef. | ExpRe. | RustS. | COUNTS |
|:-----|:-----|:-----|:-----|:-----|:-------|:-------|:-------|----:|
| x    |      |      |      |      |        |        |        | 787 |
|      | x    |      |      |      |        |        |        | 304 |
|      |      | x    |      |      |        |        |        | 163 |
|      |      |      | x    |      |        | x      | x      | 145 |
|      |      |      | x    |      |        | x      |        |  76 |
|      |      |      | x    |      |        |        |        | 201 |
|      |      |      |      | x    |        |        |        | 264 |
|      |      |      |      |      | x      |        |        | 452 |
|      |      |      |      |      |        |        | x      | 205 |
| **787**  | **304**  | **163**  | **422**  | **264**  | **452**    | **221**    | **350**    | total=*2597*

This version has two major differences to `mcds_Bukhsh`: 

* First of all, the classes `NoExposedReinforcement` and `NoRustStaining` were removed, since only a few images without Exposed Reinforcement or without Rust also had the corresposing label. It is possible to impute the missing values (i.e. for each image either `ExposedReinforcement` or  `NoExposedReinforcement` must be present) but we decided not to have "negative labels" in our dataset version (it is not common to have negative labels for a subset of labels). 
* Then we removed combinations with frequencies lower 5. For example there was one image with label `Crack` and `Scaling`. Moreover we removed remaining images with no labels, i.e. images which only had the labels `NoExposedReinforcement` and/or `NoRustStaining`.  

Notes: Contrary to what is mentioned in the Hüthwohl et al., there are images that only have the label `RustStaining` (3rd stage) without having positive labels from stage 1 or 2. After a visual inspection we decided to include these images since the label is correct. 


This dataset has 2597 images.