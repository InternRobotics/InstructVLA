import torch
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor, AutoConfig
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor
import warnings
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import pandas as pd
import string
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import math
from torchvision.transforms.functional import InterpolationMode
import re
import torch.nn.functional as F
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)
def build_transform(norm_type='imagenet'):
    if norm_type == 'imagenet':
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif norm_type == 'siglip':
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD
        
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def find_closest_aspect_ratio_v2(aspect_ratio, target_ratios, width, height, image_size):
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
    target_aspect_ratio = find_closest_aspect_ratio_v2(
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

def eagle_dynamic_preprocess(image, patch_size=14, pixel_shuffle_ratio=2, max_image_tokens=4096, min_image_tokens=16):
    orig_width, orig_height = image.size
    one_token_deserve_pixels = patch_size * pixel_shuffle_ratio
    new_width = int((orig_width+(one_token_deserve_pixels-1))//one_token_deserve_pixels * one_token_deserve_pixels)
    new_height = int((orig_height+(one_token_deserve_pixels-1))//one_token_deserve_pixels * one_token_deserve_pixels)
    image_tokens = new_width * new_height // one_token_deserve_pixels // one_token_deserve_pixels
    if image_tokens > max_image_tokens:
        resize_ratio = (image_tokens/max_image_tokens) ** 0.5
        new_width = int(new_width / resize_ratio)
        new_height = int(new_height / resize_ratio)
        new_width = int(new_width//one_token_deserve_pixels * one_token_deserve_pixels)
        new_height = int(new_height//one_token_deserve_pixels * one_token_deserve_pixels)
        image_tokens = new_width * new_height // one_token_deserve_pixels // one_token_deserve_pixels
    elif image_tokens < min_image_tokens:
        resize_ratio = (image_tokens/min_image_tokens) ** 0.5
        new_width = int(new_width / resize_ratio)
        new_height = int(new_height / resize_ratio)
        new_width = int((new_width+(one_token_deserve_pixels-1))//one_token_deserve_pixels * one_token_deserve_pixels)
        new_height = int((new_height+(one_token_deserve_pixels-1))//one_token_deserve_pixels * one_token_deserve_pixels)
        image_tokens = new_width * new_height // one_token_deserve_pixels // one_token_deserve_pixels
    
    # num2 = previous_fast_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True) * 256
    # print(f'image_tokens: {image_tokens}, num2: {num2}, orig_width: {orig_width}, orig_height: {orig_height}')
    image = image.resize((new_width, new_height))
    return image, int(image_tokens)



def pad_tensors_to_same_size(tensor_list):
    
    original_sizes = [(t.size(1), t.size(2)) for t in tensor_list]
    
    max_height = max([size[0] for size in original_sizes])
    max_width = max([size[1] for size in original_sizes])
    
    padded_tensors = []
    for tensor in tensor_list:
        h, w = tensor.size(1), tensor.size(2)
        
        pad_bottom = max_height - h
        pad_right = max_width - w
        
        padded_tensor = F.pad(tensor[None], (0, pad_right, 0, pad_bottom))
        padded_tensors.append(padded_tensor)

    padded_tensors = torch.concat(padded_tensors, dim=0)
    return padded_tensors, original_sizes

def load_image(image_file, norm_type='imagenet', upscale=False, patch_size=14, pixel_shuffle_ratio=2, max_image_tokens=4096, min_image_tokens=16):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(norm_type=norm_type)
    image, image_tokens = eagle_dynamic_preprocess(image, patch_size=patch_size, pixel_shuffle_ratio=pixel_shuffle_ratio, max_image_tokens=max_image_tokens, min_image_tokens=min_image_tokens)
    pixel_values = transform(image)
    return pixel_values, image_tokens

def split_model(model_name, device):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
        'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    print('world_size', world_size)
    num_layers_per_gpu_ = math.floor(num_layers / (world_size - 1))
    num_layers_per_gpu = [num_layers_per_gpu_] * world_size
    num_layers_per_gpu[device] = num_layers - num_layers_per_gpu_ * (world_size-1)
    print(num_layers_per_gpu)
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
    device_map[f'language_model.model.layers.{num_layers - 1}'] = device
    return device_map


# This function is used to split InternVL2-Llama3-76B
def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    return device_map


class EagleChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='OpenGVLab/InternVL-Chat-V1-5', load_in_8bit=False, version='V1.0', **kwargs):
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.36.2', 'ge')

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        device = torch.cuda.current_device()
        self.device = device
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = config.vision_config.model_type
        if model_type == 'siglip_vision_model':
            self.norm_type = 'siglip'
        else:
            self.norm_type = 'imagenet'
        print('norm_type: ', self.norm_type)
        print('model_path: ', model_path)
        print('device: ', self.device)
        if '70b' in model_path.lower():
            device_map = split_model('InternVL2-Llama3-76B', self.device)
        elif '40b' in model_path.lower():
            device_map = split_model('InternVL2-40B', self.device)
        else:
            device_map = None
            
        if device_map is not None:    
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               low_cpu_mem_usage=True,
                                               device_map=device_map, 
                                               trust_remote_code=True,
                                               load_in_8bit=load_in_8bit).eval()
        else:
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                               trust_remote_code=True,
                                               load_in_8bit=load_in_8bit).eval()
            
        if not load_in_8bit and device_map is None:
            self.model = self.model.to(device)
        self.image_size = self.model.config.vision_config.image_size

        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        if listinstr(['InternVL2-Llama3-76B'], model_path):
            device_map = split_model(model_path.split('/')[-1])
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                load_in_8bit=load_in_8bit,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=device_map).eval()
        else:
            device = torch.cuda.current_device()
            self.device = device
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=load_in_8bit).eval()
            if not load_in_8bit:
                self.model = self.model.to(device)

        self.image_size = self.model.config.vision_config.image_size
        self.version = version
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):

        if dataset is not None and listinstr(['MMDU'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        else:
            return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_video_prompt(self, prompt, dataset=None, max_nframe=64):
        for start in range(0, max_nframe, 8):
            images_to_remove = ''.join([f'<image-{i}>' for i in range(start + 1, start + 9)])
            prompt = prompt.replace(images_to_remove, '')
        for i in range(max_nframe):
            prompt = prompt.replace(f'<image-{i + 1}>', f'Frame{i + 1}')
        if listinstr(['MMBench-Video'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += '\nAnswer the question using a single word or phrase.'
        elif listinstr(['Video-MME'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += "\nAnswer with the option's letter from the given choices directly."
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if self.version == 'V1.1':
            kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=5)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        self.kwargs = kwargs_default

        if dataset is not None and listinstr(['MME'], dataset):
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['MathVista', 'MathVision'], dataset):
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        if dataset is not None and listinstr(['ChartQA_TEST', 'MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset):
            self.max_num = 24
        elif dataset is not None and listinstr(['MMBench-Video', 'Video-MME', 'Video'], dataset):
            self.max_num = 1
        else:
            self.max_num = 6

    def generate_v2(self, message, dataset=None):
        print('message: ', message)
        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    # prompt += f'<image-{image_idx}>'
                    image_idx += 1
            prompt = ' '.join([f'<image {i + 1}><image>' for i in range(image_num)]) + '\n' + prompt
          
        if listinstr(['Video'], dataset):
            prompt = self.build_video_prompt(prompt, dataset)

        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_tokens_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                # upscale_flag = False
                # print(upscale_flag, image_idx, file_name, '>1')
                curr_pixel_values, num_tokens = load_image(
                    file_name, upscale=upscale_flag, norm_type=self.norm_type, patch_size=14, pixel_shuffle_ratio=2, max_image_tokens=4096, min_image_tokens=256)
                curr_pixel_values = curr_pixel_values.cuda().to(torch.bfloat16)
                num_tokens_list.append(num_tokens)
                pixel_values_list.append(curr_pixel_values)
            # pixel_values =  torch.cat(pixel_values_list, dim=0)
            pixel_values, original_sizes = pad_tensors_to_same_size(pixel_values_list)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = listinstr(['MMMU_DEV_VAL'], dataset)
            # upscale_flag = False
            # print(upscale_flag, image_path, '1')
            pixel_values, num_tokens = load_image(
                image_path, upscale=upscale_flag, norm_type=self.norm_type, patch_size=14, pixel_shuffle_ratio=2, max_image_tokens=4096, min_image_tokens=256)
            pixel_values = pixel_values.cuda().to(torch.bfloat16)
            num_tokens_list = [num_tokens]
            original_sizes = [(pixel_values.size(1), pixel_values.size(2))]
            pixel_values = pixel_values.unsqueeze(0)
        else:
            pixel_values = None
            num_tokens_list = []
            original_sizes = []
        print('pixel_values: ', pixel_values.shape)
        print('num_tokens_list: ', num_tokens_list)
        print('original_sizes: ', original_sizes)
        print('prompt: ', prompt)
        with torch.no_grad():
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_tokens_list=num_tokens_list,
                image_original_sizes=np.array(original_sizes),
                question=prompt,
                generation_config=self.kwargs,
                verbose=False
            )
        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
 
        return self.generate_v2(message, dataset)
        

    def build_history(self, message):
        # Global Variables
        image_path = []
        image_cnt = 0

        def concat_tilist(tilist):
            nonlocal image_cnt  # Declare image_cnt as nonlocal to modify it
            prompt = ''
            for item in tilist:
                # Substitute the pattern in the text
                if item['type'] == 'text':
                    prompt += re.sub(self.pattern, self.replacement, item['value'])
                elif item['type'] == 'image':
                    image_cnt += 1
                    prompt += '<image>\n'
                    image_path.append(item['value'])
            return prompt

        # Only previous messages
        assert len(message) % 2 == 0
        history = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            history.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))

        return history, image_path, image_cnt

    def chat_inner_v2(self, message, dataset=None):

        image_cnt = 0
        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ''

        # If message is just text in the conversation
        if len(current_msg['content']) == 1 and current_msg['content'][0]['type'] == 'text':
            question = current_msg['content'][0]['value']
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per InternVL
        else:
            for msg in current_msg['content']:
                if msg['type'] == 'text':
                    question += re.sub(self.pattern, self.replacement, msg['value'])
                elif msg['type'] == 'image':
                    image_cnt += 1
                    question += '<image>\n'
                    image_path.append(msg['value'])

        if image_cnt > 1:
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=self.max_num, upscale=upscale_flag).cuda().to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_cnt == 1:
            upscale_flag = listinstr(['MMMU_DEV_VAL'], dataset)
            pixel_values = load_image(
                image_path, max_num=self.max_num, upscale=upscale_flag).cuda().to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []
        # print(question)
        response, history = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=question,
            generation_config=self.kwargs,
            history=history,
            return_history=True
        )

        response = re.sub(self.reverse_pattern, self.reverse_replacement, response)

        return response

    def chat_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        kwargs_default = dict(do_sample=False, max_new_tokens=512, top_p=None, num_beams=1)
        self.kwargs = kwargs_default
        return self.chat_inner_v2(message, dataset)

