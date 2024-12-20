<div align="center">
<h1>[ACM MM 2024] ReToMe-VA: Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack</h1>
</div>

### Share us a :star: if this repo does help

This repository is the official implementation of ***ReToMe-VA***. The paper can be accessed in [arXiv](https://arxiv.org/pdf/2408.05479) and [acm mm](https://dl.acm.org/doi/10.1145/3664647.3680959). (***Accepted by ACM MM 2024***)

## Requirements
1. Hardware Requirements
    - GPU: NVIDIA GPU with 80GB memory
2. Software Requirements
    - Python 3.9.0
    - CUDA: 11.7

    Environment:
    ```
    conda create -n retomeva python==3.9.0
    conda activate retomeva
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    pip install -r requirements
    ```

3. Datasets and Models
    - The used datasets are sampled from Kinetics-400. You need to prepare the video datasets and a csv file describing the dataset, each line per video sample. There are three items in each line: (1) video path; (2) video label and (3) clip index.
    - We use pretrained models on Kinetics-400 from [gluoncv](https://cv.gluon.ai/model_zoo/action_recognition.html) to conduct experiments, including I3D SLOW, TPN, and R(2+1)D. Change the **CONFIG_ROOT_KINETICS** of config.py into your model config path.

## Attack
```
python main.py --input_path your_data_path --test_dir output_path --retome --merge_ratio=0.5 --input_csv your_data_sample_csv_file
```

## Citation
If you find our work helpful, please leave us a star and cite our paper.

```
@inproceedings{gao2024retome,
  title={ReToMe-VA: Recursive Token Merging for Video Diffusion-based Unrestricted Adversarial Attack},
  author={Gao, Ziyi and Chen, Kai and Wei, Zhipeng and Mou, Tingshu and Chen, Jingjing and Tan, Zhiyu and Li, Hao and Jiang, Yu-Gang},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={4485--4494},
  year={2024}
}
```