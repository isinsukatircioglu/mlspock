# ML-SPOCK
This repository contains the code for the project titled "Machine Learning Supported System for Performance Assessment of Steel Structures Under Extreme Operating Conditions and Risk Management."

## Setup
* Download and install miniconda.
* Use the following commands with miniconda to create and activate your environment.
  * ```conda env create -f environment.yml```
  * ```conda activate mlspock```
## Training
You can train new networks using ```train.py```. For example:
```
python train.py --model_type=pnet2 --path_data=/mydata/mlspock/shared/mlspock_column_pc3d --mode=train --batch_size=128 --epoch=200 --segment=bottom --norm=snorm
```
You can continue training a pre-trained model using *.pth files that can be referenced using local filenames. For example:
```
python train.py --path_model=/myhome/mlspock/baselines/results/pnet2_bottom_all_snorm_2024-04-10_17-52-53 --model_type=pnet2 --path_data=/mydata/mlspock/shared/mlspock_column_pc3d --mode=train --batch_size=128 --epoch=50 --segment=bottom --norm=snorm
```
The results of each training run are saved to a newly created directory under ```~/results```.

You can evaluate a pre-trained model by ensuring that the model file is named ```best_model.pth```, and then setting the ```path_model``` option to the path where this file is located:
```
python train.py --path_model=./pretrained/pnet2_bottom_all_snorm_2024-04-29_18-33-32 --model_type=pnet2 --path_data=/mydata/mlspock/shared/mlspock_column_pc3d --mode=eval --segment=bottom --norm=snorm
```
## Test
To test the pretrained model on unseen examples, within each scenario folder, the 3D point cloud of each individual column should be saved in a separate .npy file, for instance:: ```/test_240430/W33X263-Collapse_consistent-RC90/pts029.npy (test_folder/scenario_name/p3d.npy)```.

Afterward, you can obtain the predictions by running:
```
python train.py --path_model=./pretrained/pnet2_bottom_all_snorm_2024-04-29_18-33-32 --model_type=pnet2 --path_data=/mydata/mlspock/shared/test_240430 --mode=test --segment=bottom --norm=snorm
```
## Further Information
This repository builds upon the codabase of [Pointnet, Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [Point-Transformers](https://github.com/qq456cvb/Point-Transformers).
