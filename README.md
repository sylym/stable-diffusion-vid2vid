# stable-diffusion-vid2vid

## Setup

### Clone repo

 ```bash
git clone https://github.com/sylym/stable-diffusion-vid2vid.git
cd stable-diffusion-vid2vid
 ```

### Requirements

 ```bash
conda create -n stable-diffusion-vid2vid python=3.10
conda activate stable-diffusion-vid2vid
 
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install ffmpeg
pip install -r requirements.txt
pip install -U xformers
 
cd Real-ESRGAN
python setup.py develop
cd ..
 ```

### Weights

 ```bash
cd TorchDeepDanbooru
wget https://github.com/AUTOMATIC1111/TorchDeepDanbooru/releases/download/v1/model-resnet_custom_v3.pt
cd ..
 ```

## Usage

### Write config file

Specify [parameters](https://github.com/sylym/stable-diffusion-vid2vid/blob/master/configs/vid2vid_Bocchi.yaml) such as stable diffusion model, incoming video, outgoing path, etc.

Note: 
1. The stable diffusion model needs to be diffusers format. If it is xxx.safesensors or xxx.ckpt, you need to use a [script](https://github.com/huggingface/diffusers/blob/main/scripts/convert_original_stable_diffusion_to_diffusers.py) to convert it.


```bash
# Setup Requirements
cd scripts
conda create -n stable-diffusion-vid2vid-scripts python=3.10
conda activate stable-diffusion-vid2vid-scripts
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt

# assume you have downloaded xxx.safetensors, it will out save_dir in diffusers format.
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path xxx.safetensors  --dump_path save_dir --from_safetensors

# assume you have downloaded xxx.ckpt, it will out save_dir in diffusers format.
python convert_original_stable_diffusion_to_diffusers.py --checkpoint_path xxx.ckpt  --dump_path save_dir
```
2. Inference prompt supports [Compel](https://github.com/damian0815/compel) format.


### Run

 ```bash
python vid2vid_generation.py --config ./configs/vid2vid_configs.yaml
 ```

## Thanks to
This project uses some code from [diffusers](https://github.com/huggingface/diffusers), which is licensed under Apache License 2.0; [TorchDeepDanbooru](https://github.com/AUTOMATIC1111/TorchDeepDanbooru), which is licensed under MIT License; [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), which is licensed under BSD License.

This project is based on some papers:
- [@wu2022tuneavideo] Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation
- [@zhang2023adding] Adding Conditional Control to Text-to-Image Diffusion Models


## References
```bibtex
@article{wu2022tuneavideo,
    title={Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation},
    author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
    journal={arXiv preprint arXiv:2212.11565},
    year={2022}
}

@misc{zhang2023adding,
  title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
  author={Lvmin Zhang and Maneesh Agrawala},
  year={2023},
  eprint={2302.05543},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```