# LMFD-PAD
---
## Note
This is the official repository of the paper: LMFD-PAD: Learnable Multi-level Frequency Decomposition and Hierarchical Attention Mechanism for Generalized Face Presentation Attack Detection. The paper can be found in [here](https://arxiv.org/abs/2109.07950).

## Pipeline Overview
![overview](images/workflow.png)

## Data preparation
Since the data in all used PAD datasets in our work are videos, we sample 10 frames in the average time interval of each video. In addition, the ratio of bona fide and attack is balanced by simple duplication. Finally, CSV files are generated for further training and evaluation. The format of the dataset CSV file is:
```
image_path,label
/image_dir/image_file_1.png, bonafide
/image_dir/image_file_2.png, bonafide
/image_dir/image_file_3.png, attack
/image_dir/image_file_4.png, attack
```
## Training
The training code for intra-dataset and cross-dataset experiments is same, the difference code between intra_db_main.py and cross_db_main.py is evaluation metrics.
1. Example of intra-dataset training and testing:
    ```
    python intra_db_main.py \
      --protocol_dir 'dir_containing_csv_files' \
      --backbone resnet50 \
      --pretrain True \
      --lr 0.001 \
      --batch_size 64 \
      --prefix 'custom_note' \
    ```
2. Example of cross-dataset training and testing is similar:
    ```
    python cross_db_main.py \
      --protocol_dir 'dir_containing_csv_files' \
      --backbone resnet50 \
      --pretrain True \
      --lr 0.001 \
      --batch_size 64 \
      --prefix 'custom_note' \
    ```

## Results
The results of cross-dataset evaluation under different experimental settings on four face PAD datasets. More details can be found in paper.
![cross_db](images/cross_db_results.png)

## Models
Four models pre-trained based on four cross-dataset experimental settings can be download via [google driver](https://drive.google.com/drive/folders/1qRBLkrn461r-E_Px3d_wialW-0soXEGn?usp=sharing).
Please using the following threshold for testing those pre-trained weights. The thresholds of icm_o, ocm_i, omi_c, and oci_m models are 0.7309441, 0.6971898, 0.613508, and 0.53312653, respectively.
More information and small test can be found in test.py. Please make sure give the correct model path.

if you use LMFD-HAM architecture in this repository, please cite the following paper:
```
@misc{fang2021learnable,
    title={Learnable Multi-level Frequency Decomposition and Hierarchical Attention Mechanism for Generalized Face Presentation Attack Detection},
    author={Meiling Fang and Naser Damer and Florian Kirchbuchner and Arjan Kuijper},
    year={2021},
    eprint={2109.07950},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```


## License
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt.
