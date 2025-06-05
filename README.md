# ML-SPOCK
This repository contains the code for the project titled "Machine Learning Supported System for Performance Assessment of Steel Structures Under Extreme Operating Conditions and Risk Management."
![mlspock_overview](https://github.com/user-attachments/assets/bd03fc4a-6eab-4000-b8a8-41b4e9fa01fc)

## Setup
* Download and install miniconda.
* Use the following commands with miniconda to create and activate your environment.
  * ```conda env create -f environment.yml```
  * ```conda activate mlspock```
## Training
To train a Point Transformer (PTR) on the STEEL-3dPointClouds dataset:
```
python train.py --model_type=ptr --path_data=./data/mlspock_column_pc3d --mode=train --batch_size=32 --epoch=250 --segment=bottom --norm=snorm --scaling_mode=diff --model_size=small --version=1
```
You can finetune a pre-trained model using *.pth files that can be referenced using local filenames.
To finetune our Point Transformer (PTR) on the STEEL-3dPointClouds dataset:
```
python train.py --model_type=ptr --path_data=./data/mlspock_column_pc3d --mode=train --batch_size=32 --epoch=250 --segment=bottom --norm=snorm --scaling_mode=diff --model_size=small --version=1 --resume --path_model=./pretrained/ptr_small_segment[bottom]_snorm[diff_scale]_input[3d]_lprot[all]_batch_32_lr_0.0005_optimizer_AdamW_seed42_v1
```
To finetune our scenario-based classification model PTR+Cls[All] on the STEEL-3dPointClouds dataset:
```
python train.py --model_type=ptr_classify --path_data=./data/mlspock_column_pc3d --mode=train --batch_size=32 --epoch=250 --segment=bottom --norm=snorm --scaling_mode=diff --model_size=small --classify=all --split=late --version=1 --resume --path_model=./pretrained/ptr_classify_small_segment[bottom]_snorm[diff_scale]_input[3d]_lprot[all]_batch_32_lr_0.0005_optimizer_AdamW_classify[all_late]_seed42_v1
```
To finetune our scenario-guided model PTR+[LR, BC, LP, CS] on the STEEL-3dPointClouds dataset:
```
python train.py --model_type=ptr --path_data=./data/mlspock_column_pc3d --mode=train --batch_size=32 --epoch=250 --segment=bottom --norm=snorm --scaling_mode=diff --model_size=small --lratio_known --boundary_known --lprotocol_known --dimension_known --version=1 --resume --path_model=./pretrained/ptr_small_segment[bottom]_snorm[diff_scale]_input[3d_lratio_boundary_lprot_colsize]_lprot[all]_batch_32_lr_0.0005_optimizer_AdamW_seed42_v1
```
The results of each training run are saved to a newly created directory under ```~/experiments```.

You can evaluate a pre-trained model by ensuring that the model file is named ```best_model.pth```, and by setting the ```path_model``` option to the path where this file is located:
```
python train.py --model_type=ptr --path_data=./data/mlspock_column_pc3d --mode=eval --batch_size=32 --segment=bottom --norm=snorm --scaling_mode=diff --model_size=small --version=1 --path_model=./pretrained/ptr_small_segment[bottom]_snorm[diff_scale]_input[3d]_lprot[all]_batch_32_lr_0.0005_optimizer_AdamW_seed42_v1
```
## Test
To test a pre-trained model on unseen examples, each scenario folder should contain the 3D point cloud of each individual column saved as a separate .npy file. For example: ```/test_240430/W33X263-Collapse_consistent-RC90/pts029.npy (test_folder/scenario_name/p3d.npy)```.
You can then obtain predictions by running:
```
python train.py --model_type=ptr --path_data=./data/test_240430 --mode=test  --segment=bottom --norm=snorm --scaling_mode=diff --model_size=small --version=1 --path_model=./pretrained/ptr_small_segment[bottom]_snorm[diff_scale]_input[3d]_lprot[all]_batch_32_lr_0.0005_optimizer_AdamW_seed42_v1 --gt
```
or
```
python train.py --model_type=ptr_classify --path_data=./data/test_240430 --mode=test  --segment=bottom --norm=snorm --scaling_mode=diff --model_size=small --version=1 --path_model=./pretrained/ptr_classify_small_segment[bottom]_snorm[diff_scale]_input[3d]_lprot[all]_batch_32_lr_0.0005_optimizer_AdamW_classify[all_late]_seed42_v1 --classify=all --split=late --gt
```

## Further Information
This repository builds upon the codabase of [Pointnet, Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [Point-Transformers](https://github.com/qq456cvb/Point-Transformers).
