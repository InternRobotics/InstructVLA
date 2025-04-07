import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Sequence, Optional

import torch

import transformers
from torch.utils.data import Dataset


from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from vla.eagle_utils import EagleProcessor
from types import SimpleNamespace
import os.path as osp
import numpy as np

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
        '/mnt/petrelfs/yangshuai1/yangshuai1/share_mllm/Eagle2-2B',
        max_input_tiles=1,
        model_spec=SimpleNamespace(
            num_image_token = 256,
            template = "qwen2-chat"
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained('/mnt/petrelfs/yangshuai1/yangshuai1/share_mllm/Eagle2-2B', 
                                                use_fast=True,
                                                trust_remote_code=True)
    
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path='/mnt/petrelfs/yangshuai1/rep/cogact_with_history/bunny_dataset/finetune/bunny_allava_1.3m.json',
                                          processor=processor)
    from IPython import embed;embed()
    train_dataset[0]

