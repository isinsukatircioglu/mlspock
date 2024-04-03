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
python train.py --model_type=ptr --path_data=/mydata/mlspock/shared/mlspock_column_pc3d --mode=train --batch_size=128 --epoch=1 --segment=bottom --norm=snorm
```
The results of each training run are saved to a newly created directory under ```~/results```.
## Further Information
This repository builds upon the codabase of [Pointnet, Pointnet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) and [Point-Transformers](https://github.com/qq456cvb/Point-Transformers).
