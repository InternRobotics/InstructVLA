# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import base64
import os
from io import BytesIO
from typing import List, Union

import numpy as np
import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.
"""

import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple, Union

IGNORE_INDEX = -100

class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    FALCON_CHAT = auto()
    CHATGLM3 = auto()
    INTERNVL_ZH = auto()
    MPT = auto()
    LLAMA3 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM:
            # source: https://huggingface.co/THUDM/chatglm-6b/blob/1d240ba371910e9282298d4592532d7f0f3e9f3e/modeling_chatglm.py#L1302-L1308
            # source2: https://huggingface.co/THUDM/chatglm2-6b/blob/e186c891cf64310ac66ef10a87e6635fa6c2a579/modeling_chatglm.py#L926
            round_add_n = 1 if self.name == "chatglm2" else 0
            if system_prompt:
                ret = system_prompt + self.sep
            else:
                ret = ""

            for i, (role, message) in enumerate(self.messages):
                if i % 2 == 0:
                    ret += f"[Round {i//2 + round_add_n}]{self.sep}"

                if message:
                    ret += f"{role}：{message}{self.sep}"
                else:
                    ret += f"{role}："
            return ret
        elif self.sep_style == SeparatorStyle.CHATML:
            ret = "" if system_prompt == "" else system_prompt + self.sep + "\n"
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + message + self.sep + "\n"
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.CHATGLM3:
            ret = ""
            if self.system_message:
                ret += system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + "\n" + " " + message
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.CHATINTERN:
            # source: https://huggingface.co/internlm/internlm-chat-7b-8k/blob/bd546fa984b4b0b86958f56bf37f94aa75ab8831/modeling_internlm.py#L771
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                # if i % 2 == 0:
                #     ret += "<s>"
                if message:
                    ret += role + ":" + message + seps[i % 2] + "\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = system_prompt
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = system_prompt
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.FALCON_CHAT:
            ret = ""
            if self.system_message:
                ret += system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"

            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL_ZH:
            seps = [self.sep, self.sep2]
            ret = self.system_message + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA3:
            ret = system_prompt + self.sep
            for role, message in self.messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# Note that for inference, using the Hermes-2 and internlm2-chat templates is equivalent.
register_conv_template(
    Conversation(
        name="Hermes-2",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。人工智能实验室致力于原始技术创新，开源开放，共享共创，推动科技进步和产业发展。',
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_token_ids=[
            2,
            6,
            7,
            8,
        ],
        stop_str="<|endoftext|>",
    )
)


register_conv_template(
    Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。人工智能实验室致力于原始技术创新，开源开放，共享共创，推动科技进步和产业发展。',
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_token_ids=[2, 92543, 92542],
    )
)


register_conv_template(
    Conversation(
        name="phi3-chat",
        system_template="<|system|>\n{system_message}",
        # note: The new system prompt was not used here to avoid changes in benchmark performance.
        # system_message='我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。人工智能实验室致力于原始技术创新，开源开放，共享共创，推动科技进步和产业发展。',
        system_message="你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。",
        roles=("<|user|>\n", "<|assistant|>\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|end|>",
        stop_token_ids=[2, 32000, 32007],
    )
)
register_conv_template(
    Conversation(
        name="llama3-chat",
        system_template="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}",
        system_message="You are an AI assistant whose name is Eagle-Next.",
        roles=(
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ),
        sep_style=SeparatorStyle.LLAMA3,
        sep="<|eot_id|>",
        stop_token_ids=[128259, 128001],
    )
)

# Qwen-chat default template
# source: https://huggingface.co/Qwen/Qwen-7B-Chat/blob/main/qwen_generation_utils.py#L130
register_conv_template(
    Conversation(
        name="qwen2-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message="You are a helpful assistant.",
        roles=("<|im_start|>user", "<|im_start|>assistant"),
        sep_style=SeparatorStyle.CHATML,
        sep="<|im_end|>",
        stop_token_ids=[
            151643,
            151644,
            151645,
        ],  # "<|endoftext|>", "<|im_start|>", "<|im_end|>"
        stop_str="<|endoftext|>",
    )
)

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
    prefix = "This is a video:\n"
    for i in range(num_frames):
        if time_position:
            frame_txt = f"Frame {i+1} sampled at {meta_list[i]:.2f} seconds: <image>\n"
        else:
            frame_txt = f"Frame {i+1}: <image>\n"
        prefix += frame_txt
    return prefix


def load_video(video_path, num_frames=64, frame_cache_root=None):
    if isinstance(video_path, str):
        # video = decord.VideoReader(video_path)
        video = None
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
        if "disk_path" in image:
            return Image.open(image["disk_path"])
        elif "base64" in image:
            return Image.open(BytesIO(base64.b64decode(image["base64"])))
        elif "url" in image:
            response = requests.get(image["url"])
            return Image.open(BytesIO(response.content))
        elif "bytes" in image:
            return Image.open(BytesIO(image["bytes"]))
        elif "np_array" in image:
            return Image.fromarray(image["np_array"])
        else:
            raise ValueError(f"Invalid image: {image}")
    else:
        raise ValueError(f"Invalid image: {image}")


def build_transform(input_size, norm_type="imagenet"):
    if norm_type == "imagenet":
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    elif norm_type == "siglip":
        MEAN, STD = SIGLIP_MEAN, SIGLIP_STD

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio_v2(aspect_ratio, target_ratios, width, height, image_size):
    """
    previous version mainly foucs on ratio.
    We also consider area ratio here.
    """
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        abs(aspect_ratio - target_aspect_ratio)
        (ratio[0] * ratio[1] * image_size * image_size) / area
        """
        new area > 60% of original image area is enough.
        """
        factor_based_on_area_n_ratio = min(
            (ratio[0] * ratio[1] * image_size * image_size) / area, 0.6
        ) * min(target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio)

        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio_v2(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

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
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class ModelSpecificValues:
    def __init__(self, template, num_image_token):
        self.template = template
        self.num_image_token = num_image_token


def prepare(
    model_spec,
    system_message,
    tokenizer,
    pixel_values,
    question,
    history=None,
    num_patches_list=None,
    IMG_START_TOKEN="<img>",
    IMG_END_TOKEN="</img>",
    IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
    llm_only=False,
):
    if history is None and pixel_values is not None and "<image>" not in question:
        question = "<image>\n" + question

    if num_patches_list is None:
        num_patches_list = [1] * pixel_values.shape[0] if pixel_values is not None else []
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    template = get_conv_template(model_spec.template)
    template.system_message = system_message

    history = [] if history is None else history
    for old_question, old_answer in history:
        template.append_message(template.roles[0], old_question)
        template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = (
            IMG_START_TOKEN
            + IMG_CONTEXT_TOKEN * model_spec.num_image_token * num_patches
            + IMG_END_TOKEN
        )
        if llm_only:
            query = query.replace("<image>", "", 1)
        else:
            query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")

    return (
        pixel_values,
        model_inputs["input_ids"],
        model_inputs["attention_mask"],
    )


class EagleProcessor:
    def __init__(
        self,
        model_path: Union[str, None] = None,
        model_spec: Union[ModelSpecificValues, None] = None,
        max_input_tiles: int = 1,
        use_local_eagle_hg_model: bool = True,
    ):
        # This defaults use local eagle hg model card
        if model_path.endswith("/"):
            model_path = model_path[:-1]

        if model_spec is None:
            model_spec = ModelSpecificValues(
                template="qwen2-chat",
                num_image_token=64,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        tokens_to_keep = ["<box>", "</box>", "<ref>", "</ref>"]
        tokenizer.additional_special_tokens = [
            item for item in tokenizer.additional_special_tokens if item not in tokens_to_keep
        ]
        self.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = config.vision_config.model_type
        if model_type == "siglip_vision_model":
            self.norm_type = "siglip"
        elif model_type == "MOB":
            self.norm_type = "siglip"
        else:
            self.norm_type = "imagenet"
        self.config = config
        self.image_size = config.force_image_size
        self.context_len = tokenizer.model_max_length
        self.per_tile_len = 256
        self.model_spec = model_spec
        self.max_input_tiles = max_input_tiles
        self.tokenizer.padding_side = "left"

    def scale_image_size_by(self, factor):
        self.image_size = int(self.image_size * factor)
        self.model_spec.num_image_token = int(self.model_spec.num_image_token * factor**2)
        print(
            f"New image size: {self.image_size}, New num_image_token: {self.model_spec.num_image_token}"
        )

    def get_img_context_token(self, IMG_CONTEXT_TOKEN="<IMG_CONTEXT>"):
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        return img_context_token_id

    def get_eos_token_id(self):
        template = get_conv_template(self.model_spec.template)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
        return eos_token_id

    def prepare_input(self, params):
        system_message = params["prompt"][0]["content"]
        send_messages = params["prompt"][1:]
        max_input_tiles = self.max_input_tiles
        video_frame_num = params.get("video_frame_num", 64)

        global_image_cnt = 0
        history, pil_images, max_input_tile_list = [], [], []
        for message in send_messages:
            if message["role"] == "user":
                prefix = ""
                if "image" in message:
                    for image_data in message["image"]:
                        pil_images.append(load_image(image_data))
                        prefix = prefix + f"<image {global_image_cnt + 1}><image>\n"
                        global_image_cnt += 1
                        max_input_tile_list.append(max_input_tiles)
                if "video" in message:
                    raise Exception("Not support video now, decord causes issues.")
                    for video_data in message["video"]:
                        video_frames, tmp_prefix = load_video(
                            video_data, num_frames=video_frame_num
                        )
                        pil_images.extend(video_frames)
                        prefix = prefix + tmp_prefix
                        global_image_cnt += len(video_frames)
                        max_input_tile_list.extend([1] * len(video_frames))
                content = prefix + message["content"]
                history.append(
                    [
                        content,
                    ]
                )
            else:
                history[-1].append(message["content"])
        question, history = history[-1][0], history[:-1]

        if global_image_cnt == 1:
            question = question.replace("<image 1><image>\n", "<image>\n")
            history = [
                [item[0].replace("<image 1><image>\n", "<image>\n"), item[1]] for item in history
            ]

        assert len(max_input_tile_list) == len(
            pil_images
        ), "The number of max_input_tile_list and pil_images should be the same."

        transform = build_transform(input_size=self.image_size, norm_type=self.norm_type)
        if len(pil_images) > 0:
            max_input_tiles_limited_by_contect = self.max_input_tiles
            while True:
                image_tiles = []
                for current_max_input_tiles, pil_image in zip(max_input_tile_list, pil_images):
                    if self.config.dynamic_image_size:
                        tiles = dynamic_preprocess(
                            pil_image,
                            image_size=self.image_size,
                            max_num=min(
                                current_max_input_tiles, max_input_tiles_limited_by_contect
                            ),
                            use_thumbnail=self.config.use_thumbnail,
                        )
                    else:
                        tiles = [pil_image]
                    image_tiles += tiles
                if len(image_tiles) * self.per_tile_len < self.context_len:
                    break
                else:
                    max_input_tiles_limited_by_contect -= 2

                if max_input_tiles_limited_by_contect < 1:
                    break

            pixel_values = [transform(item) for item in image_tiles]
            pixel_values = torch.stack(pixel_values).to(dtype=torch.bfloat16)
        else:
            pixel_values = None

        (
            pixel_values,
            input_ids,
            attention_mask,
        ) = prepare(
            model_spec=self.model_spec,
            system_message=system_message,
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=question,
            history=history,
        )
        data = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return data
    
    def preprocess_inputs_and_labels(
        self, params,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        ) -> Dict:

        system_message = params["prompt"][0]["content"]
        send_messages = params["prompt"][1:]
        max_input_tiles = self.max_input_tiles
        video_frame_num = params.get("video_frame_num", 64)

        global_image_cnt = 0
        history, pil_images, max_input_tile_list = [], [], []
        for message in send_messages:
            if message["role"] == "user":
                prefix = ""
                if "image" in message:
                    for image_data in message["image"]:
                        pil_images.append(load_image(image_data))
                        prefix = prefix + f"<image {global_image_cnt + 1}><image>\n"
                        global_image_cnt += 1
                        max_input_tile_list.append(max_input_tiles)
                content = prefix + message["content"]
                history.append(
                    [
                        content,
                    ]
                )
            else:
                history[-1].append(message["content"])

        if global_image_cnt == 1:
            history = [
                [item[0].replace("<image 1><image>\n", "<image>\n"), item[1]] for item in history
            ]

        assert len(max_input_tile_list) == len(
            pil_images
        ), "The number of max_input_tile_list and pil_images should be the same."

        transform = build_transform(input_size=self.image_size, norm_type=self.norm_type)
        if len(pil_images) > 0:
            max_input_tiles_limited_by_contect = self.max_input_tiles
            while True:
                image_tiles = []
                for current_max_input_tiles, pil_image in zip(max_input_tile_list, pil_images):
                    if self.config.dynamic_image_size:
                        tiles = dynamic_preprocess(
                            pil_image,
                            image_size=self.image_size,
                            max_num=min(
                                current_max_input_tiles, max_input_tiles_limited_by_contect
                            ),
                            use_thumbnail=self.config.use_thumbnail,
                        )
                    else:
                        tiles = [pil_image]
                    image_tiles += tiles
                if len(image_tiles) * self.per_tile_len < self.context_len:
                    break
                else:
                    max_input_tiles_limited_by_contect -= 2

                if max_input_tiles_limited_by_contect < 1:
                    break

            pixel_values = [transform(item) for item in image_tiles]
            pixel_values = torch.stack(pixel_values).to(dtype=torch.bfloat16)
        else:
            pixel_values = None

        template = get_conv_template(self.model_spec.template)

        for question, answer in history:
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], answer)

        query = template.get_prompt()


        num_patches_list = [1] * pixel_values.shape[0] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.model_spec.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        
        tokens = self.tokenizer(query, return_tensors="pt")
        input_ids = tokens['input_ids'][0]

        # Define the tokens for "assistant\n" and "<|im_end|>"
        assistant_tokens = self.tokenizer.encode("assistant\n")  # This will give [77091, 198]
        end_token = self.tokenizer.encode("<|im_end|>")[0]

        # Initialize the labels with -100 (to mask out)
        labels = input_ids.clone()
        mask = torch.ones_like(input_ids, dtype=bool)

        idx = 0
        while idx < input_ids.shape[0] - 2:
            if labels[idx] == 77091 and labels[idx+1] == 198:
                label_len = 0
                while input_ids[label_len + idx + 2] != 151645:
                    label_len+=1
                mask[idx+2: label_len + idx + 2 + 1] = 0 

                idx = label_len + idx + 2 + 1
            else:
                idx+=1

        labels[mask] = IGNORE_INDEX

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
        )

    def post_process(self, generation_output):
        all_responses = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return all_responses

    def collate_fn(self, all_examples):
        pixel_values_list = [ex["pixel_values"] for ex in all_examples]
        input_ids_list = [ex["input_ids"] for ex in all_examples]
        attention_mask_list = [ex["attention_mask"] for ex in all_examples]

        assert isinstance(pixel_values_list, List)
        assert isinstance(input_ids_list, List)
        assert isinstance(attention_mask_list, List)

        pixel_values = torch.cat(pixel_values_list, dim=0)

        tokenized_batch = {
            "input_ids": [ip[0] for ip in input_ids_list],
            "attention_mask": [am[0] for am in attention_mask_list],
        }

        # Apply left padding
        padded_batch = self.tokenizer.pad(
            tokenized_batch,
            padding=True,  # Ensures padding to max sequence length
            return_tensors="pt",  # Convert to PyTorch tensors
        )

        input_ids = padded_batch.input_ids
        attention_mask = padded_batch.attention_mask
        data = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return BatchFeature(data)


def reshape_model_embeddings(model, factor):
    module = model.vision_model.vision_model.embeddings
    num_pos = module.num_positions * factor**2
    curr_dtype = module.position_ids.dtype
    curr_device = module.position_ids.device
    values = torch.arange(num_pos, dtype=curr_dtype, device=curr_device).expand((1, -1))

    module.register_buffer("position_ids", values, persistent=False)

    # curr_len = module.position_ids.shape[1]
    # new_len = int(curr_len * factor ** 2)
    # module.position_ids = module.position_ids[:, :new_len]
    print(f"Reshaped position_ids to {num_pos}")


def get_embeddings(
    self,
    pixel_values=None,
    input_ids=None,
    attention_mask=None,
    visual_features=None,
    output_hidden_states=None,
) -> torch.LongTensor:
    assert self.img_context_token_id is not None
    assert pixel_values is not None
    if visual_features is not None:
        vit_embeds = visual_features
    else:
        vit_embeds = self.extract_feature(pixel_values)

    input_embeds = self.language_model.get_input_embeddings()(input_ids)
    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    input_ids = input_ids.reshape(B * N)
    selected = input_ids == self.img_context_token_id
    assert selected.sum() != 0
    input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

    input_embeds = input_embeds.reshape(B, N, C)

    # return hidden_states
    embeddings = self.language_model.forward(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    embeddings = embeddings.hidden_states[-1]
    return embeddings