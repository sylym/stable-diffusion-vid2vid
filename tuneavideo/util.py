import os
import shutil
import numpy as np
from typing import Union
from PIL import Image

import torch

from tqdm import tqdm
from einops import rearrange
from safetensors.torch import load_file
import cv2
import moviepy.editor as mp
import moviepy.video.fx.all as vfx
import subprocess
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from controlnet_aux import HEDdetector, MLSDdetector, OpenposeDetector
from .controlnet_utils import ade_palette


def save_videos_grid(videos: torch.Tensor, path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    num = 0
    for x in videos:
        image = x.squeeze().numpy()
        image = np.transpose(image[[2, 1, 0], :, :], (1, 2, 0))
        image = (image * 65535.0).round().astype(np.uint16)
        cv2.imwrite(os.path.join(path, f"{os.path.basename(path)}_{num}.png"), image)
        num += 1


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents


def use_lora(pretrained_LoRA_path, pipe, alpha):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    state_dict = load_file(pretrained_LoRA_path)

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:

        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipe.text_encoder
        else:
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipe.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)
    return pipe


def down_up_sample(temp_path_mp4, down_sample, no_down=False):
    video = cv2.VideoCapture(temp_path_mp4)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    video.release()
    if down_sample == 2 or down_sample == 4:
        if no_down:
            print("up sample with x" + str(down_sample))
            temp_path_mp4_folder = os.path.dirname(temp_path_mp4)
            subprocess.run(["python", "./Real-ESRGAN/inference_realesrgan_video.py", "-i", temp_path_mp4, "-o", temp_path_mp4_folder, "-s", str(down_sample)], check=True)
            os.remove(temp_path_mp4)
            video_path = temp_path_mp4[:-4] + "_out.mp4"
            os.rename(os.path.abspath(video_path), os.path.abspath(temp_path_mp4))
        else:
            print("down and up sample with x" + str(down_sample))
            temp_path_mp4_up_res_folder = os.path.dirname(temp_path_mp4)
            subprocess.run(["python", "./Real-ESRGAN/inference_realesrgan_video.py", "-i", temp_path_mp4, "-o", temp_path_mp4_up_res_folder, "-s", str(down_sample)], check=True)
            os.remove(temp_path_mp4)
            video_path = temp_path_mp4[:-4] + "_out.mp4"
            subprocess.run(["ffmpeg", "-i", video_path, "-vf", f"scale={width}:{height}", os.path.abspath(temp_path_mp4)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.remove(video_path)
    else:
        print("do noting with down_up_sample")


def video_clip(video_path, start_frame, end_frame, output_path):
    clip = mp.VideoFileClip(video_path)
    start_time = start_frame / clip.fps
    end_time = (end_frame + start_frame) / clip.fps
    clip = clip.subclip(start_time, end_time)
    clip.write_videofile(output_path, fps=clip.fps, logger=None)


def get_video_fps(video_path):
    clip = mp.VideoFileClip(video_path)
    return clip.fps


def get_video_frame_count(video_path):
    clip = mp.VideoFileClip(video_path)
    return clip.reader.nframes


def merge_video(video_list, output_path, speed=1, generate_video=True):
    video_list_out = []
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    for video_path in video_list:  # 遍历视频路径列表
        for image_path in os.listdir(video_path):
            shutil.copyfile
            image_path
        video = mp.VideoFileClip(str(video_path))  # 把每个视频文件转换成VideoFileClip对象
        video_list_out.append(video)  # 添加到视频列表中
    final_video = mp.concatenate_videoclips(video_list_out)  # 合并视频列表
    final_video = final_video.fx(vfx.speedx, 1 / speed)
    final_video.write_videofile(output_file)  # 保存合并后的视频


def controlnet_image_preprocessing(image_list, video_prepare_type):
    image_list_out = []
    if video_prepare_type == "canny":
        for image in image_list:
            image = np.array(image)
            low_threshold = 100
            high_threshold = 200
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            image_list_out.append(image)

    elif video_prepare_type == "depth":
        depth_estimator = pipeline('depth-estimation')
        for image in image_list:
            image = depth_estimator(image)['depth']
            image = np.array(image)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            image = Image.fromarray(image)
            image_list_out.append(image)

    elif video_prepare_type == "mlsd":
        mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = mlsd(image)
            image_list_out.append(image)

    elif video_prepare_type == "hed":
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = hed(image)
            image_list_out.append(image)

    elif video_prepare_type == "normal":
        depth_estimator = pipeline("depth-estimation", model="Intel/dpt-hybrid-midas")
        for image in image_list:
            image = image.convert("RGB")
            image = depth_estimator(image)['predicted_depth'][0]
            image = image.numpy()
            image_depth = image.copy()
            image_depth -= np.min(image_depth)
            image_depth /= np.max(image_depth)
            bg_threhold = 0.4
            x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
            x[image_depth < bg_threhold] = 0
            y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
            y[image_depth < bg_threhold] = 0
            z = np.ones_like(x) * np.pi * 2.0
            image = np.stack([x, y, z], axis=2)
            image /= np.sum(image ** 2.0, axis=2, keepdims=True) ** 0.5
            image = (image * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
            image = Image.fromarray(image)
            image_list_out.append(image)

    elif video_prepare_type == "openpose":
        openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = openpose(image)
            image_list_out.append(image)

    elif video_prepare_type == "scribble":
        hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
        for image in image_list:
            image = hed(image, scribble=True)
            image_list_out.append(image)

    elif video_prepare_type == "seg":
        image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
        image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
        for image in image_list:
            image = image.convert('RGB')
            pixel_values = image_processor(image, return_tensors="pt").pixel_values
            with torch.no_grad():
                outputs = image_segmentor(pixel_values)
            seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
            color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
            palette = np.array(ade_palette())
            for label, color in enumerate(palette):
                color_seg[seg == label, :] = color
            color_seg = color_seg.astype(np.uint8)
            image = Image.fromarray(color_seg)
            image_list_out.append(image)

    else:
        image_list_out = image_list

    return image_list_out