from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.util import save_videos_grid, use_lora
from diffusers.schedulers import DDIMScheduler
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.loaders import MultiTokenCLIPTokenizer
import argparse
import torch
from omegaconf import OmegaConf
from accelerate.utils import set_seed
from tuneavideo.models.controlnet import ControlNetModel
import os
from compel import Compel

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="./configs/Continuous_frame_challenge.yaml")
parser.add_argument("--cuda", type=str, default="0")
args = parser.parse_args()
gpu_id = args.cuda
args = OmegaConf.load(args.config)

weight_dtype = torch.float32
if args.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif args.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16

if args.seed is not None:
    set_seed(args.seed)

scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
if os.path.exists(args.pretrained_controlnet_path):
    controlnet = ControlNetModel.from_pretrained(args.pretrained_controlnet_path)
else:
    controlnet = ControlNetModel.from_pretrained("lllyasviel/" + os.path.basename(args.pretrained_controlnet_path))
tokenizer = MultiTokenCLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder='tokenizer')
text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(args.pretrained_vae_path, subfolder='vae')
unet = UNet3DConditionModel.from_pretrained(args.output_dir, subfolder='unet')
pipe = TuneAVideoPipeline(scheduler=scheduler, unet=unet, vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, controlnet=controlnet)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

# Use pretrained_embedding
if args.inference_data.pretrained_embedding.use_pretrained_embedding:
    all_pretrained_embedding_dict = {}
    for pretrained_embedding_dict in args.inference_data.pretrained_embedding.pretrained_embedding_data:
        all_pretrained_embedding_dict[pretrained_embedding_dict["pretrained_embedding_prompt"]] = pretrained_embedding_dict["pretrained_embedding_path"]
    pipe.load_textual_inversion_embeddings(all_pretrained_embedding_dict)

# Use LoRA
if args.inference_data.LoRA.use_LoRA:
    for lora_dict in args.inference_data.LoRA.LoRA_data:
        pipe = use_lora(lora_dict["pretrained_LoRA_path"], pipe, lora_dict["LoRA_alpha"])

pipe = pipe.to(weight_dtype).to(f"cuda:{gpu_id}")

# main pipeline
compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
embeds = compel.build_conditioning_tensor(args.inference_data.prompt)
negative_embeds = compel.build_conditioning_tensor(args.inference_data.negative_prompt)
args.inference_data.prompt = None
args.inference_data.negative_prompt = None

generator = torch.Generator(device=f"cuda:{gpu_id}")
if args.seed is not None:
    generator.manual_seed(args.seed)

sample = pipe(generator=generator, scheduler_path=args.pretrained_model_path, prompt_embeds=embeds, negative_prompt_embeds=negative_embeds, **args.inference_data).videos
save_videos_grid(sample, args.inference_data.video_output_path)
