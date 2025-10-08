<p align="center">
  <h1 align="center">Expressive Human Model Tracker </h1>
<p align="center">

<div align="center"> 
  <img src="assets/Docs/visual.gif">
</div>

## 📌 Introduction
This repository contains the EHM tracking implementation associated with the paper ["GUAVA: Generalizable Upper Body 3D Gaussian Avatar"](https://github.com/Pixel-Talk/GUAVA).

The tracker estimates an expressive human mesh model from front-facing upper-body video or images, directly supporting GUAVA's training and inference workflows.

## 🛠️ Setup

This guide outlines the steps to set up and run the project components, which have been tested on Ubuntu Linux 20.04.

#### Cloning the Repository
```shell
# Via SSH
git clone git@github.com:Pixel-Talk/EHM-Trakcer.git
or
# Via HTTPS 
git clone https://github.com/Pixel-Talk/EHM-Trakcer.git

cd EHM-Trakcer
```

### Environment Setup
Our default, provided install method is based on Conda package and environment management:

**The environment dependencies are identical to the updated GUAVA environment.**
```shell
# Create and Activate Conda Environment:
conda create --name EHM-Trakcer python=3.10
conda activate EHM-Trakcer

# Install Core Dependencies:
pip install -r requirements.txt

# Install PyTorch3D:
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
```

## 📦 Model Preparation

#### Parametric Models:
**Configure `assets/smplx` and `assets/flame` as in the [GUAVA repository](https://github.com/Pixel-Talk/GUAVA).**
- SMPLX: Download `SMPLX_NEUTRAL_2020.npz` from [SMPLX](https://smpl-x.is.tue.mpg.de/download.php) and place it in the `assets/SMPLX`.
- FLAME: Download the `generic_model.pkl` from [FLAME2020](https://flame.is.tue.mpg.de/download.php). Save this file to both `assets/FLAME/FLAME2020/generic_model.pkl` and `assets/SMPLX/flame_generic_model.pkl`.

#### Example videos or images:
- Manual Download:
Example videos and images for testing:
Download the [example](https://drive.google.com/file/d/1clwXCV92T5YtAXsI7k2pqf40PnvLmF_t/view?usp=sharing), unzip the file, and place it in the `assets/examples`.

- Command-line Download:
  ```shell
  bash assets/Docs/run_download_examples.sh
  ```

#### Pretrained model:
To run tracking, you will need to acquire the following:

- Manual Download:
Download the [pretrained weight](https://drive.google.com/file/d/1g_4YKQvLSWo8yzYHgNstr91RCD4rne8p/view?usp=sharing), unzip the file, and place it in the `pretrained`.

- Command-line Download:
  ```shell
  bash assets/Docs/run_download_pretrained.sh
  ```

## 🚀 Running Scripts
**Note:** To ensure **accurate tracking**, we discard frames with low hand keypoint confidence scores and those where the two hands are in close proximity.
The specific filtering logic can be reviewed in [data_prepare_pipeline.py](src/data_prepare_pipeline.py).

Execute the following command to perform **Video Tracking**:
```shell
export PYTHONPATH='.'
python tracking_video.py \
    --in_root assets/examples/videos \
    --output_dir results/example_video  \
    --save_vis_video --save_images \
    --check_hand_score 0.0 -n 1 -v 0
```

Execute the following command to perform **Image Tracking**:
```shell
export PYTHONPATH='.'
CUDA_VISIBLE_DEVICES=0 python -m src.tracking_single_image \
    --source_dir assets/examples/images \
    --output_dir results/example_image  \
    --save_vis_video --save_visual_render
```

Building training dataset for GUAVA:
```shell
export PYTHONPATH='.'
# Run the tracking script
# 8 threads for 4 gpus
python tracking_video.py \
 --in_root /path/to/video_dataset \
  --output_dir  /path/to/tracked_video \
 --check_hand_score 0.45 \
 --tracking_with_interval \
 -n 8 -v 0,1,2,3

# Combine Tracking Dataset
python -m src.build_lmdb_dataset \
 --data_folders /path/to/tracked_video \
 --save_path /path/to/combined_dataset

# Split into Training and Validation Sets
python -m src.split_dataset \
 --data_path /path/to/combined_dataset \
 --num_valid 100
```


## 📖 BibTeX
If you find our work helpful, please ⭐ our repository and cite:
```bibtex
@article{GUAVA,
  title={GUAVA: Generalizable Upper Body 3D Gaussian Avatar},
  author={Zhang, Dongbin and Liu, Yunfei and Lin, Lijian and Zhu, Ye and Li, Yang and Qin, Minghan and Li, Yu and Wang, Haoqian},
  journal={arXiv preprint arXiv:2505.03351},
  year={2025}
}