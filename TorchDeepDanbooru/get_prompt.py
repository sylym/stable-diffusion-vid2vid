from PIL import Image
import numpy as np
import torch
import shutil
from .deep_danbooru_model import DeepDanbooruModel
from tqdm import tqdm
import cv2
import os


def frames2time(frames, fps):  # 给定fps，将帧数转化为指定格式的视频时间（匹配popsub，小时数是假的2333，不能超过1小时，想要什么格式可以自己改ww）
    fpm = fps * 60
    a = int(frames / fpm)  # 分钟数（取整)
    frames %= fpm  # 余下帧数
    b = frames / fps
    return "0_" + f"{str(a):0>2}" + "_" + f"{b:0>5.2f}"


def get_prompt(prompt_theme, temp_path_out, probability_threshold):
    model = DeepDanbooruModel()
    model.load_state_dict(torch.load('./TorchDeepDanbooru/model-resnet_custom_v3.pt'))

    model.eval()
    model.half()
    model.cuda()
    prompt_dict = {}
    filenames = os.listdir(temp_path_out)
    for filename in tqdm(filenames):
        if not filename.endswith(".png"):
            continue
        pic = Image.open(temp_path_out + "/" + filename).convert("RGB").resize((512, 512))
        a = np.expand_dims(np.array(pic, dtype=np.float32), 0) / 255

        with torch.no_grad(), torch.autocast("cuda"):
            x = torch.from_numpy(a).cuda()

            # first run
            y = model(x)[0].detach().cpu().numpy()

            # measure performance
            for n in range(10):
                model(x)

        for i, p in enumerate(y):
            key = model.tags[i].replace('_', ' ')
            if key == 'rating:safe' or key in prompt_theme:
                continue
            if p < probability_threshold:
                continue
            if key not in prompt_dict:
                prompt_dict[key] = p
            else:
                prompt_dict[key] += p

    return prompt_dict


def video_to_frames(video_path):
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        folder_name = video_path + file_name
        os.makedirs(folder_name, exist_ok=True)
        vc = cv2.VideoCapture(video_path+'/'+video_name)
        c=0
        rval=vc.isOpened()

        while rval:
            c = c + 1
            rval, frame = vc.read()
            pic_path = folder_name+'/'
            if rval:
                cv2.imwrite(pic_path + str(c) + '.png', frame)
                cv2.waitKey(1)
            else:
                break
        vc.release()
    return pic_path


def detect_prompt(video_path_in, start_frame, end_frame, prompt_theme, prompt_num, prompt_head, probability_threshold):
    shutil.rmtree("./temp/DeepDanboorutemp", ignore_errors=True)
    os.makedirs("./temp/DeepDanboorutemp")
    for i in range(start_frame, end_frame):
        shutil.copy(os.path.join(video_path_in, os.listdir(video_path_in)[i]), f"./temp/DeepDanboorutemp/{os.listdir(video_path_in)[i]}")
    prompt_pre = get_prompt(prompt_theme, "./temp/DeepDanboorutemp", probability_threshold)
    shutil.rmtree("./temp/DeepDanboorutemp", ignore_errors=True)

    #os.makedirs(os.path.dirname(promot_path_out), exist_ok=True)
    temp_prompt_pre_sorted = sorted(prompt_pre.items(), key = lambda kv:(kv[1], kv[0]))
    prompt_pre_sorted = temp_prompt_pre_sorted[::-1]
    prompt_pre_out = prompt_pre_sorted[0:prompt_num+1]
    prompt_out = prompt_head + ","
    for prompt_key_value in prompt_pre_out:
        prompt_out += prompt_key_value[0] + ","
    prompt_out += prompt_theme
    #f = open(promot_path_out, 'w', encoding='utf-8')
    #f.writelines(prompt_out)
    return prompt_out

