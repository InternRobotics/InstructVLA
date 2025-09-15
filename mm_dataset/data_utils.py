import os
import copy
from dataclasses import dataclass, field
import json
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any, Sequence

import torch

import transformers
import datasets
from torch.utils.data import Dataset


from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from vla.eagle_utils import EagleProcessor
from types import SimpleNamespace
import os.path as osp
import numpy as np
import random

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful assistant."

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: AutoTokenizer,
                 processor: AutoProcessor,
                 image_path = './bunny_dataset/finetune/images',
                 ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.image_path = image_path

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.processor = processor

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            has_image=('image' in sources)
            conversation = sources['conversations']
            formatted_conv = [{"role": "system", "content": DEFAULT_SYSTEM_MESSAGE}]
            for turn in conversation:
                if turn['from'] == 'human':
                    if "<image>\n" in turn['value'] and has_image:
                        formatted_conv.append({
                            "role": "user",
                            "content": turn['value'].replace("<image>\n",""),
                            "image": [{'np_array': np.asarray(Image.open(osp.join(self.image_path, sources['image'])))}]
                        })
                    else:
                        formatted_conv.append({
                            "role": "user",
                            "content": turn['value'].replace("<image>\n","")
                        })
                elif turn['from'] == 'gpt':
                    formatted_conv.append({
                            "role": "assistant",
                            "content": turn['value']
                        })
            inputs = self.processor.preprocess_inputs_and_labels({"prompt": formatted_conv})
            return inputs
        except Exception as e:
            if i>1:
                return self.__getitem__(i-1)
            else:
                return self.__getitem__(i+999)

class LazyPointingDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    grounding_hints = [
        " Please include grounding annotations using <points> tags to describe key regions.",
        " Describe the image with <points> annotations marking important visual elements.",
        " Add structured grounding using <points> tags for notable objects or areas in the image.",
        " Use <points> tags to highlight and explain key image regions in your response.",
        " Include spatial annotations with <points> tags to describe what the image shows.",
        " Mark relevant visual features using <points> annotations in your description.",
        " Ground your description with <points> tags referencing specific image regions.",
        " Use <points> to identify and explain visual entities mentioned in the image.",
        " Provide detailed grounding by tagging image parts with <points> and descriptions.",
        " In your answer, include <points> tags to associate content with image regions.",
        " Add <points> tags to describe objects.",
        " Include <points> for visual grounding.",
        " Ground your answer with <points>.",
        " Highlight areas with <points> tags.",
        " Insert <points> for important spots.",
    ]

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 processor: AutoProcessor,
                 data_path:str = '/mnt/petrelfs/yangshuai1/yangshuai1/datasets/pixmo-point-explanations-images/',
                 ):
        super(LazyPointingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = datasets.load_dataset(data_path,split='train')
        self.processor = processor

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            has_image=('image' in sources)
            grounding_hint = random.choice(self.grounding_hints)
            formatted_conv = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": sources["question"] + grounding_hint,
                    "image": [{'np_array': np.asarray(sources["image"])}]
                },
                {
                    "role": "assistant",
                    "content": sources['response']
                }
            ]

            inputs = self.processor.preprocess_inputs_and_labels({"prompt": formatted_conv})
            inputs["pixel_values"] = inputs["pixel_values"]
            return inputs
        except Exception as e:
            if i>1:
                return self.__getitem__(i-1)
            else:
                return self.__getitem__(i+999)
        

class LazyPointDetectionDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    grounding_hints = [
        "Count the {label} in the image, then point to them.",
        "How many {label} are there? Point to them.",
        "Count every {label} in the picture, then point to them.",
        "Locate the {label} and count them, then point to them.",
        "Find all the {label}. How many are there? Point to them.",
        "Find each {label}. How many are there? Point to them.",
        "Point to and count the {label} in the picture.",
    ]

    def __init__(self,
                 tokenizer: AutoTokenizer,
                 processor: AutoProcessor,
                 annotation_path :str = '/mnt/petrelfs/yangshuai1/yangshuai1/datasets/pixmo-points-local/annotations_filtered_10_points.json',
                 image_dir: str = '/mnt/petrelfs/yangshuai1/yangshuai1/datasets/pixmo-points-local'
                 ):
        super(LazyPointDetectionDataset, self).__init__()
        self.tokenizer = tokenizer

        with open(annotation_path,'r') as f:
            self.annotation = json.load(f)
        self.image_dir = image_dir
        self.processor = processor

    def points_to_text(self, points, label_text, alt_text):
        # Round and sort the points
        processed_points = [[round(p["x"], 1), round(p["y"], 1)] for p in points]
        processed_points.sort(key=lambda p: p[0] * 10000 + p[1])

        if len(processed_points) == 1:
            x, y = processed_points[0]
            return f'<point x="{x:.1f}" y="{y:.1f}" alt="{label_text}">{alt_text}</point>'
        else:
            point_text = []
            for ix, (x, y) in enumerate(processed_points, start=1):
                point_text.append(f'x{ix}="{x:.1f}"')
                point_text.append(f'y{ix}="{y:.1f}"')
            point_text_str = " ".join(point_text)
            return f'<points {point_text_str} alt="{alt_text}">{label_text}</points>'

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.annotation[i]
            # some images are transparent, so we always convert them into RGBA and add white background
            image = Image.open(osp.join(self.image_dir, sources["image_path"])).convert("RGBA")

            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image).convert("RGB")

            label = sources["label"]
            grounding_hint = random.choice(self.grounding_hints).format(label=label.lower())
            user_prompt = grounding_hint

            if not sources.get("points"):
                assistant_target = "There are none."
            else:
                label_text = str(label).lower()
                count = sources.get("count")
                alt_text = f"{count} {label_text}" if count is not None else label_text
                assistant_target = self.points_to_text(sources["points"], label_text, alt_text)

            formatted_conv = [
                {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": user_prompt,
                    "image": [{'np_array': np.asarray(image)}]
                },
                {
                    "role": "assistant",
                    "content": assistant_target
                }
            ]

            inputs = self.processor.preprocess_inputs_and_labels({"prompt": formatted_conv})
            inputs["pixel_values"] = inputs["pixel_values"]
            return inputs

        except Exception as e:
            if i>1:
                return self.__getitem__(i-1)
            else:
                return self.__getitem__(i+999)


IGNORE_INDEX = -100
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    model_max_length:int = 1024

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))


        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX)

        input_ids = input_ids[:, :self.model_max_length]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        labels = labels[:, :self.model_max_length]

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        image_flags = torch.zeros((input_ids.shape[0],1))

        if 'pixel_values' in instances[0]:
            images = [instance['pixel_values'] for instance in instances]
            images_not_none = []
            for idx, image in enumerate(images):
                if image is not None:
                    image_flags[idx] = 1
                    images_not_none.append(image)
                else:
                    # the image is just a place holder
                    images_not_none.append(torch.zeros(( 1, 3 ,448, 448))) 
        
        batch['image_flags'] = image_flags

        if all(x.shape == images_not_none[0].shape for x in images_not_none):
            batch['pixel_values'] = torch.stack(images_not_none)
                
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                processor: AutoProcessor) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          processor=processor)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator) 

if __name__ == "__main__":


    processor = EagleProcessor(
        'ckpt/Eagle2-2B',
        max_input_tiles=1,
        model_spec=SimpleNamespace(
            num_image_token = 256,
            template = "qwen2-chat"
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained('ckpt/Eagle2-2B', 
                                                use_fast=True,
                                                trust_remote_code=True)
    
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path='bunny_dataset/finetune/bunny_allava_1.3m.json',
                                          processor=processor)

    # train_dataset = LazyPointDetectionDataset(tokenizer=tokenizer,
    #                                     processor=processor)

    from IPython import embed;embed()
    item = train_dataset[0]

