import argparse
import os.path
import subprocess
from omegaconf import OmegaConf
import threading
from TorchDeepDanbooru.get_prompt import detect_prompt
from tuneavideo.util import get_video_fps, merge_video, down_up_sample, get_video_frame_count
import shutil
import math
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'


def generate_configs():
    for train_config_num in range(math.ceil(args.n_sample_frames / args.num_splits)):
        train_configs = dict(seed=args.seed,
                             mixed_precision=args.mixed_precision,
                             use_8bit_adam=args.use_8bit_adam,
                             enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
                             pretrained_model_path=args.pretrained_model_path,
                             pretrained_vae_path=args.pretrained_vae_path,
                             max_train_steps=args.num_train_steps,
                             output_dir=f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{train_config_num}")
        train_configs_train_data = dict(video_path=args.video_in_path,
                                        prompt=args.train_prompt,
                                        sample_frame_rate=args.sample_frame_rate)
        if args.hiresfix and args.hiresfix_raise:
            if args.video_width % 2 != 0 or args.video_height % 2 != 0:
                raise ValueError("video_width and video_height must be divisible by 2")
            train_configs_train_data["width"] = args.video_width // 2
            train_configs_train_data["height"] = args.video_height // 2
        else:
            train_configs_train_data["width"] = args.video_width
            train_configs_train_data["height"] = args.video_height
        if args.n_sample_frames % args.num_splits != 0 and train_config_num == math.ceil(args.n_sample_frames / args.num_splits) - 1:
            train_configs_train_data["n_sample_frames"] = args.n_sample_frames % args.num_splits
        else:
            train_configs_train_data["n_sample_frames"] = args.num_splits
        train_configs_train_data["sample_start_idx"] = train_config_num * args.num_splits * args.sample_frame_rate
        train_configs["train_data"] = train_configs_train_data
        OmegaConf.save(train_configs, f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{train_config_num}.yaml")

    for inference_config_num in range(math.ceil(args.n_sample_frames / args.num_splits)):
        inference_configs = dict(pretrained_model_path=args.pretrained_model_path,
                                 pretrained_vae_path=args.pretrained_vae_path,
                                 pretrained_controlnet_path=f"./checkpoints/controlnet/sd-controlnet-{args.controlnet_video_prepare_type}",
                                 output_dir=f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{inference_config_num}",
                                 seed=args.seed, mixed_precision=args.mixed_precision,)
        inference_config_inference_data = dict(negative_prompt=args.inference_negative_prompt,
                                               video_output_path=f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{inference_config_num}_pic",
                                               num_inference_steps=args.num_inference_steps,
                                               guidance_scale=args.guidance_scale,
                                               use_inv_latent=args.use_inv_latent,
                                               num_inv_steps=args.num_inv_steps,
                                               ddim_prompt=args.ddim_prompt,
                                               use_vid2vid=args.use_vid2vid,
                                               video_strength=args.video_strength,
                                               use_conrolnet=args.use_conrolnet,
                                               video_prepare_type=args.controlnet_video_prepare_type,
                                               controlnet_conditioning_scale=args.controlnet_conditioning_scale)
        if args.hiresfix and args.hiresfix_raise:
            inference_config_inference_data["width"] = args.video_width // 2
            inference_config_inference_data["height"] = args.video_height // 2
        else:
            inference_config_inference_data["width"] = args.video_width
            inference_config_inference_data["height"] = args.video_height
        if args.n_sample_frames % args.num_splits != 0 and inference_config_num == math.ceil(args.n_sample_frames / args.num_splits) - 1:
            inference_config_inference_data["video_length"] = args.n_sample_frames % args.num_splits
        else:
            inference_config_inference_data["video_length"] = args.num_splits
        inference_config_inference_data["sample_start_idx"] = inference_config_num * args.num_splits * args.sample_frame_rate
        inference_config_inference_data["sample_frame_rate"] = args.sample_frame_rate
        inference_config_inference_data["video_input_path"] = args.video_in_path
        inference_config_inference_data["LoRA"] = args.LoRA
        inference_config_inference_data["pretrained_embedding"] = args.pretrained_embedding
        if args.prompt_generation:
            inference_config_inference_data["prompt"] = detect_prompt(video_path_in=args.video_in_path, start_frame=inference_config_inference_data["sample_start_idx"],
                  end_frame=inference_config_inference_data["video_length"], prompt_theme="", probability_threshold=args.prompt_generation_probability_threshold,
                  prompt_num=args.prompt_generation_num, prompt_head=args.inference_prompt)
        else:
            inference_config_inference_data["prompt"] = args.inference_prompt
        inference_configs["inference_data"] = inference_config_inference_data
        OmegaConf.save(inference_configs, f"./temp/inference_{os.path.basename(args.video_in_path).split('.')[0]}_{inference_config_num}.yaml")


def generate_hiresfix_configs():
    for hiresfix_train_config_num in range(math.ceil(frame_count / args.hiresfix_num_splits)):
        hiresfix_train_configs = dict(seed=args.seed,
                                      mixed_precision=args.mixed_precision,
                                      use_8bit_adam=args.use_8bit_adam,
                                      enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
                                      pretrained_model_path=args.pretrained_model_path,
                                      pretrained_vae_path=args.pretrained_vae_path,
                                      max_train_steps=args.hiresfix_num_train_steps,
                                      output_dir=f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{hiresfix_train_config_num}")
        hiresfix_train_configs_train_data = dict(
            video_path=f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full.mp4",
            prompt=args.train_prompt,
            sample_frame_rate=1,
            width=args.video_width,
            height=args.video_height)
        if frame_count % args.hiresfix_num_splits != 0 and hiresfix_train_config_num == math.ceil(frame_count / args.hiresfix_num_splits) - 1:
            hiresfix_train_configs_train_data["n_sample_frames"] = frame_count % args.hiresfix_num_splits
        else:
            hiresfix_train_configs_train_data["n_sample_frames"] = args.hiresfix_num_splits
        hiresfix_train_configs_train_data["sample_start_idx"] = hiresfix_train_config_num * args.hiresfix_num_splits
        hiresfix_train_configs["train_data"] = hiresfix_train_configs_train_data
        OmegaConf.save(hiresfix_train_configs, f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{hiresfix_train_config_num}.yaml")

    for hiresfix_inference_configs_num in range(math.ceil(frame_count / args.hiresfix_num_splits)):
        hiresfix_inference_configs = dict(pretrained_model_path=args.pretrained_model_path,
                                          pretrained_vae_path=args.pretrained_vae_path,
                                          pretrained_controlnet_path=f"./checkpoints/controlnet/sd-controlnet-{args.hiresfix_controlnet_video_prepare_type}",
                                          output_dir=f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{hiresfix_inference_configs_num}",
                                          seed=args.seed, mixed_precision=args.mixed_precision,)
        hiresfix_inference_config_inference_data = dict(negative_prompt=args.inference_negative_prompt,
                                                        video_output_path=f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{hiresfix_inference_configs_num}_pic",
                                                        num_inference_steps=args.hiresfix_num_inference_steps,
                                                        guidance_scale=args.hiresfix_guidance_scale,
                                                        use_inv_latent=args.hiresfix_use_inv_latent,
                                                        num_inv_steps=args.hiresfix_num_inv_steps,
                                                        ddim_prompt=args.ddim_prompt,
                                                        use_vid2vid=True,
                                                        video_strength=args.hiresfix_strength,
                                                        width=args.video_width,
                                                        use_conrolnet=args.hiresfix_use_conrolnet,
                                                        video_prepare_type=args.hiresfix_controlnet_video_prepare_type,
                                                        controlnet_conditioning_scale=args.hiresfix_controlnet_conditioning_scale,
                                                        height=args.video_height)
        if frame_count % args.hiresfix_num_splits != 0 and hiresfix_inference_configs_num == math.ceil(frame_count / args.hiresfix_num_splits) - 1:
            hiresfix_inference_config_inference_data["video_length"] = frame_count % args.hiresfix_num_splits
        else:
            hiresfix_inference_config_inference_data["video_length"] = args.hiresfix_num_splits
        hiresfix_inference_config_inference_data["sample_start_idx"] = hiresfix_inference_configs_num * args.hiresfix_num_splits
        hiresfix_inference_config_inference_data["sample_frame_rate"] = 1
        hiresfix_inference_config_inference_data["video_input_path"] = f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full.mp4"
        hiresfix_inference_config_inference_data["LoRA"] = args.LoRA
        hiresfix_inference_config_inference_data["pretrained_embedding"] = args.pretrained_embedding
        if args.prompt_generation:
            hiresfix_inference_config_inference_data["prompt"] = detect_prompt(video_path_in=args.video_in_path, probability_threshold=args.prompt_generation_probability_threshold,
                                                                  start_frame=hiresfix_inference_config_inference_data["sample_start_idx"],
                                                                  end_frame=hiresfix_inference_config_inference_data["video_length"],
                                                                  prompt_theme="", prompt_num=args.prompt_generation_num, prompt_head=args.inference_prompt)
        else:
            hiresfix_inference_config_inference_data["prompt"] = args.inference_prompt
        hiresfix_inference_configs["inference_data"] = hiresfix_inference_config_inference_data
        OmegaConf.save(hiresfix_inference_configs, f"./temp/hiresfix_inference_{os.path.basename(args.video_in_path).split('.')[0]}_{hiresfix_inference_configs_num}.yaml")


def hiresfix_train_inference(gpu_id, num):
    if gpu_id == args.gpu_ids.split(",")[0]:
        if not args.use_hiresfix_pretraining_model:
            subprocess.run(["python", "train_tuneavideo.py", "--cuda", gpu_id, "--config", f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], check=True)
        subprocess.run(["python", "inference.py", "--cuda", gpu_id, "--config", f"./temp/hiresfix_inference_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], check=True)
    else:
        if not args.use_hiresfix_pretraining_model:
            subprocess.run(["python", "train_tuneavideo.py", "--cuda", gpu_id, "--config", f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["python", "inference.py", "--cuda", gpu_id, "--config", f"./temp/hiresfix_inference_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if args.delete_checkpoints:
        shutil.rmtree(f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{num}")


def train_inference(gpu_id, num):
    if gpu_id == args.gpu_ids.split(",")[0]:
        if not args.use_pretraining_mode:
            subprocess.run(["python", "train_tuneavideo.py", "--cuda", gpu_id, "--config", f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], check=True)
        subprocess.run(["python", "inference.py", "--cuda", gpu_id, "--config", f"./temp/inference_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], check=True)
    else:
        if not args.use_pretraining_mode:
            subprocess.run(["python", "train_tuneavideo.py", "--cuda", gpu_id, "--config", f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["python", "inference.py", "--cuda", gpu_id, "--config", f"./temp/inference_{os.path.basename(args.video_in_path).split('.')[0]}_{num}.yaml"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    if args.delete_checkpoints:
        shutil.rmtree(f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{num}")


def split_chunks(m, n):
    string = [chunk for chunk in range(m)]
    return [string[chunk:chunk+n] for chunk in range(0, len(string), n)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/vid2vid_badapple.yaml")
    args = parser.parse_args()
    args = OmegaConf.load(args.config)

    os.makedirs("./temp/", exist_ok=True)
    os.makedirs(os.path.dirname(args.video_out_path), exist_ok=True)

    # 生成训练和推理配置文件
    if not args.only_hiresfix:
        generate_configs()

    video_list = []
    train_inference_threads = []

    # 生成多线程处理块
    num_splits_gpu_list = split_chunks(math.ceil(args.n_sample_frames / args.num_splits), len(args.gpu_ids.split(",")))

    # 训练推理线程
    for i in range(len(num_splits_gpu_list)):
        for ii in range(len(num_splits_gpu_list[i])):
            if not args.only_hiresfix:
                gpu_ids = args.gpu_ids.split(",")
                cudaid = gpu_ids[ii]
                train_inference_thread = threading.Thread(target=train_inference, args=(cudaid, num_splits_gpu_list[i][ii]))
                train_inference_thread.start()
                train_inference_threads.append(train_inference_thread)
            video_list.append(os.path.abspath(f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_{num_splits_gpu_list[i][ii]}_pic"))
        for train_inference_thread in train_inference_threads:
            train_inference_thread.join()

    if args.hiresfix:
        # 图片汇总
        merge_video(video_list, f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full_pic_processed", generate_video=False)

        # 超分辨率
        if args.hiresfix_raise:
            down_up_sample(f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full_pic_processed", down_sample=2, no_down=True)
        else:
            down_up_sample(f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full_pic_processed", down_sample=2, no_down=False)

        # 生成视频
        subprocess.run(["ffmpeg", "-y", "-r", str(get_video_fps(args.video_in_path)), "-f", "image2", "-i", f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full_pic_processed/%d.png", "-c:v", "libx264", "-crf", "0", "-vf", f"fps={str(get_video_fps(args.video_in_path))}", f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full.mp4"], stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)

        for num_hiresfix in range(args.hiresfix_num):
            # 生成hiresfix训练和推理配置文件
            frame_count = get_video_frame_count(f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full.mp4")
            print("inference frame count: ", frame_count)
            generate_hiresfix_configs()

            hiresfi_video_list = []
            hiresfix_train_inference_threads = []

            # 生成hiresfix多线程处理块
            hiresfix_num_splits_gpu_list = split_chunks(math.ceil(frame_count / args.hiresfix_num_splits), len(args.gpu_ids.split(",")))

            # hiresfix训练推理线程
            for i in range(len(hiresfix_num_splits_gpu_list)):
                for ii in range(len(hiresfix_num_splits_gpu_list[i])):
                    gpu_ids = args.gpu_ids.split(",")
                    cudaid = gpu_ids[ii]
                    hiresfix_train_inference_thread = threading.Thread(target=hiresfix_train_inference, args=(cudaid, hiresfix_num_splits_gpu_list[i][ii]))
                    hiresfix_train_inference_thread.start()
                    hiresfix_train_inference_threads.append(hiresfix_train_inference_thread)
                    hiresfi_video_list.append(os.path.abspath(f"./temp/hiresfix_train_{os.path.basename(args.video_in_path).split('.')[0]}_{hiresfix_num_splits_gpu_list[i][ii]}_pic"))
                for hiresfix_train_inference_thread in hiresfix_train_inference_threads:
                    hiresfix_train_inference_thread.join()

            if num_hiresfix == args.hiresfix_num - 1:
                # 生成输出视频
                merge_video(hiresfi_video_list, args.video_out_path, speed=args.sample_frame_rate, fps=get_video_fps(args.video_in_path))
            else:
                # 生成视频用于下一次hiresfix
                merge_video(hiresfi_video_list, f"./temp/train_{os.path.basename(args.video_in_path).split('.')[0]}_full", fps=get_video_fps(args.video_in_path))
            if not args.hiresfix_with_train:
                args.use_hiresfix_pretraining_model = True
    else:
        # 生成输出视频
        merge_video(video_list, args.video_out_path, speed=args.sample_frame_rate, fps=get_video_fps(args.video_in_path))

    if args.delete_temp:
        shutil.rmtree("./temp/")
