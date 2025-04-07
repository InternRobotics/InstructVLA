
"""
A model worker executes the model.
"""
from transformers import AutoModel, AutoTokenizer, TextIteratorStreamer, AutoConfig
import argparse
import base64
import json
import os
import decord
import threading
import time
from io import BytesIO
from threading import Thread
import math
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


def get_seq_frames(total_num_frames, desired_num_frames=-1, stride=-1):
    """
    Calculate the indices of frames to extract from a video.

    Parameters:
    total_num_frames (int): Total number of frames in the video.
    desired_num_frames (int): Desired number of frames to extract.

    Returns:
    list: List of indices of frames to extract.
    """
    
    assert desired_num_frames > 0 or stride > 0 and not (desired_num_frames > 0 and stride > 0)

    if stride > 0:
        return list(range(0, total_num_frames, stride))
    
    # Calculate the size of each segment from which a frame will be extracted
    seg_size = float(total_num_frames - 1) / desired_num_frames

    seq = []
    for i in range(desired_num_frames):
        # Calculate the start and end indices of each segment
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        # Append the middle index of the segment to the list
        seq.append((start + end) // 2)

    return seq

def build_video_prompt(meta_list, num_frames, time_position=False):
    # if time_position is True, the frame_timestamp is used.
    # 1. pass time_position, 2. use env TIME_POSITION
    time_position = os.environ.get("TIME_POSITION", time_position)
    prefix = f"This is a video:\n"
    for i in range(num_frames):
        if time_position:
            frame_txt = f"Frame {i+1} sampled at {meta_list[i]:.2f} seconds: <image>\n"
        else:
            frame_txt = f"Frame {i+1}: <image>\n"
        prefix += frame_txt
    return prefix

def load_video(video_path, num_frames=64, frame_cache_root=None):
    if isinstance(video_path, str):
        video = decord.VideoReader(video_path)
    elif isinstance(video_path, dict):
        assert False, 'we not support vidoe: "video_path" as input'
    fps = video.get_avg_fps()
    sampled_frames = get_seq_frames(len(video), num_frames)
    samepld_timestamps = [i / fps for i in sampled_frames]
    frames = video.get_batch(sampled_frames).asnumpy()
    images = [Image.fromarray(frame) for frame in frames]
    
    return images, build_video_prompt(samepld_timestamps, len(images), time_position=True)

def load_image(image):
    if isinstance(image, str) and os.path.exists(image):
        return Image.open(image)
    elif isinstance(image, dict):
        if 'disk_path' in image:
            return Image.open(image['disk_path'])
        elif 'base64' in image:
            return Image.open(BytesIO(base64.b64decode(image['base64'])))
        elif 'url' in image:
            response = requests.get(image['url'])
            return Image.open(BytesIO(response.content))
        elif 'bytes' in image:
            return Image.open(BytesIO(image['bytes']))
        else:
            raise ValueError(f'Invalid image: {image}')
    else:
        raise ValueError(f'Invalid image: {image}')

def build_transform(input_size, norm_type='imagenet'):
    if norm_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif norm_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
        
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    previous version mainly foucs on ratio.
    We also consider area ratio here.
    """
    best_factor = float('-inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        area_ratio = (ratio[0]*ratio[1]*image_size*image_size)/ area
        """
        new area > 60% of original image area is enough.
        """
        factor_based_on_area_n_ratio = min((ratio[0]*ratio[1]*image_size*image_size)/ area, 0.6)* \
                                     min(target_aspect_ratio/aspect_ratio, aspect_ratio/target_aspect_ratio)
        
        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio
        
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def split_model(model_path, device):

    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers

    num_layers_per_gpu_ = math.floor(num_layers / (world_size - 1))
    num_layers_per_gpu = [num_layers_per_gpu_] * world_size
    num_layers_per_gpu[device] = num_layers - num_layers_per_gpu_ * (world_size-1)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = device
    device_map['mlp1'] = device
    device_map['language_model.model.tok_embeddings'] = device
    device_map['language_model.model.embed_tokens'] = device
    device_map['language_model.output'] = device
    device_map['language_model.model.norm'] = device
    device_map['language_model.lm_head'] = device
    device_map['language_model.model.rotary_emb'] = device
    device_map[f'language_model.model.layers.{num_layers - 1}'] = device
    return device_map

class ModelWorker:
    def __init__(self, model_path, model_name,
                 load_8bit, device):

        if model_path.endswith('/'):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split('/')
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + '_' + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        print(f'Loading the model {self.model_name}')

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        tokens_to_keep = ['<box>', '</box>', '<ref>', '</ref>']
        tokenizer.additional_special_tokens = [item for item in tokenizer.additional_special_tokens if item not in tokens_to_keep]
        self.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = config.vision_config.model_type
        self.device = torch.cuda.current_device()
        if model_type == 'siglip_vision_model':
            self.norm_type = 'siglip'
        elif model_type == 'MOB':
            self.norm_type = 'siglip'
        else:
            self.norm_type = 'imagenet'
        print('norm_type: ', self.norm_type)
        if any(x in model_path.lower() for x in ['34b']):
            device_map = split_model(model_path, self.device)
        else:
            device_map = None

        if device_map is not None:    
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True,
                                               device_map=device_map, 
                                               trust_remote_code=True,
                                               load_in_8bit=load_8bit).eval()
        else:
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True,
                                               load_in_8bit=load_8bit).eval()  
        if not load_8bit and device_map is None:
            self.model = self.model.to(device)
        self.load_8bit = load_8bit
        
        self.model_path = model_path
        self.image_size = self.model.config.force_image_size
        self.context_len = tokenizer.model_max_length
        self.per_tile_len = 256
        print(self.model)
    def reload_model(self):
        del self.model
        torch.cuda.empty_cache()
        if self.device == 'auto':
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            # This can make distributed deployment work properly
            self.model = AutoModel.from_pretrained(
                self.model_path,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                trust_remote_code=True).eval()
        else:
            self.model = AutoModel.from_pretrained(
                self.model_path,
                load_in_8bit=self.load_8bit,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).eval()
        if not self.load_8bit and not self.device == 'auto':
            self.model = self.model.cuda()

    @torch.inference_mode()
    def generate(self, params):
        system_message = params['prompt'][0]['content']
        send_messages = params['prompt'][1:]
        max_input_tiles = params['max_input_tiles']
        temperature = params['temperature']
        top_p = params['top_p']
        max_new_tokens = params['max_new_tokens']
        repetition_penalty = params['repetition_penalty']
        video_frame_num = params.get('video_frame_num', 64)
        do_sample = True if temperature > 0.0 else False

        global_image_cnt = 0
        history, pil_images, max_input_tile_list = [], [], []
        
        for message in send_messages:
            if message['role'] == 'user':
                prefix = ''
                if 'image' in message:
                    for image_data in message['image']:
                        pil_images.append(load_image(image_data))
                        prefix = prefix + f'<image {global_image_cnt + 1}><image>\n'
                        global_image_cnt += 1
                        max_input_tile_list.append(max_input_tiles)
                if 'video' in message:
                    for video_data in message['video']:
                        video_frames, tmp_prefix = load_video(video_data, num_frames=video_frame_num)
                        pil_images.extend(video_frames)
                        prefix = prefix + tmp_prefix
                        global_image_cnt += len(video_frames)
                        max_input_tile_list.extend([1] * len(video_frames))
                content = prefix + message['content']
                history.append([content, ])
            else:
                history[-1].append(message['content'])
        question, history = history[-1][0], history[:-1]

        if global_image_cnt == 1:
            question = question.replace('<image 1><image>\n', '<image>\n')
            history = [[item[0].replace('<image 1><image>\n', '<image>\n'), item[1]] for item in history]


        try:
            assert len(max_input_tile_list) == len(pil_images), 'The number of max_input_tile_list and pil_images should be the same.'
        except Exception as e:
            from IPython import embed; embed()
            exit()
            print(f'Error: {e}')
            print(f'max_input_tile_list: {max_input_tile_list}, pil_images: {pil_images}')
            # raise e

        old_system_message = self.model.system_message
        self.model.system_message = system_message
        
        transform = build_transform(input_size=self.image_size, norm_type=self.norm_type)
        if len(pil_images) > 0:
            max_input_tiles_limited_by_contect = params['max_input_tiles']
            while True:
                image_tiles = []
                num_patches_list = []
                for current_max_input_tiles, pil_image in zip(max_input_tile_list, pil_images):
                    if self.model.config.dynamic_image_size:
                        tiles = dynamic_preprocess(
                            pil_image, image_size=self.image_size, max_num=min(current_max_input_tiles, max_input_tiles_limited_by_contect),
                            use_thumbnail=self.model.config.use_thumbnail)
                    else:
                        tiles = [pil_image]
                    num_patches_list.append(len(tiles))
                    image_tiles += tiles
                if (len(image_tiles) * self.per_tile_len < self.context_len):
                    break
                else:
                    max_input_tiles_limited_by_contect -= 2
                
                if max_input_tiles_limited_by_contect < 1:
                    break
                    
            pixel_values = [transform(item) for item in image_tiles]

            
            pixel_values = torch.stack(pixel_values).to(self.model.device, dtype=torch.bfloat16)
            
        else:
            pixel_values = None

        generation_config = dict(
            num_beams=1,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_length=self.context_len,
            top_p=top_p,
        )
        print(f'pixel_values: {pixel_values.shape}')
        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            history=history,
            return_history=False,
            num_patches_list=num_patches_list,
            generation_config=generation_config,
        )
        self.model.system_message = old_system_message
        return {'text': response, 'error_code': 0}





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='/home/zhidingy/workspace/eagle-next/internvl_chat/work_dirs/release/test/Eagle2-2B')
    parser.add_argument('--model-name', type=str, default='Eagle2')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--load-8bit', action='store_true')
    args = parser.parse_args()
    print(f'args: {args}')

    worker = ModelWorker(
                         args.model_path,
                         args.model_name,
                         args.load_8bit,
                         args.device)
    prompt = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'Describe these two images in details respectively.', 
            'image':[
                {'url': 'https://www.nvidia.com/content/dam/en-zz/Solutions/about-nvidia/logo-and-brand/01-nvidia-logo-vert-500x200-2c50-d@2x.png'},
                {'url': "https://www.google.com.hk/images/branding/googlelogo/2x/googlelogo_color_272x92dp.png"}
            ]
        }
    ]
    params = {
        'prompt': prompt,
        'max_input_tiles': 24,
        'temperature': 0.7,
        'top_p': 1.0,
        'max_new_tokens': 4096,
        'repetition_penalty': 1.0,
    }
    print(worker.generate(params))