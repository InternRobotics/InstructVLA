import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor
import warnings
from PIL import Image
import argparse

import pandas as pd
import string
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import math
from torchvision.transforms.functional import InterpolationMode
import re
import os.path as osp
import json
# Not in this dir
from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load, load_vla
from PIL import Image
from tqdm import tqdm
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class InstructVLA():
    def __init__(self, model_path='TBD',
                        work_dir='TBD',
                        **kwargs):
        assert model_path is not None

        self.json_path = "./mm_evaluation/language_evaluation_with_gt.json"
        self.json = json.load(open(self.json_path))
        self.image_path = "./mm_evaluation/language_evaluation"
        self.work_dir = work_dir

        self.caption, self.qa, self.instruction = [], [], []

        for sample in self.json:
            self.caption.append(sample['annotation']["Caption"])
            self.qa.append((sample['annotation']["QA"])[0] )
            self.instruction.append((sample['annotation']["CR"])[0] )
            self.instruction.append((sample['annotation']["CC"])[0] )
        print(f'number of the test cases: Caption {len(self.caption)}; QA: {len(self.qa)}, Instruction: {len(self.instruction)}')

        self.model_path = model_path
        self.load_model()

    def load_model(self):
        device = torch.cuda.current_device()
        self.device = device
        self.model = load_vla(self.model_path, stage = "stage2").eval().to(torch.bfloat16)
        self.tokenizer = self.model.tokenizer
        device_map = None
        
        self.model = self.model.to(device)

    @torch.inference_mode()
    def generate_one_sample(self, image: Image, Instruction: str):
        messages = [
            {"content":"You are a helpful assistant."}, # system
            {
                "role": "user",
                "content": Instruction,
                "image":[{'np_array': np.asarray(image)}]
            }
        ]
        inputs = self.model.processor.prepare_input(dict(prompt=messages))
        autocast_dtype = torch.bfloat16
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
            output = self.model.vlm.generate(
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                pixel_values=inputs['pixel_values'].cuda(),
                max_length=350,
                output_hidden_states=False,
                return_dict_in_generate=False,
                stopping_criteria=self.model.stopping_criteria
            )
        response = self.tokenizer.decode(output[0]).replace("<new_token_0>","").replace("<|im_end|>","")
        torch.cuda.empty_cache()
        return response

    def evaluate_captioning(self):

        result = []
        count = 1 
        for case in tqdm(self.json):
            response = self.generate_one_sample(
                Image.open(osp.join(self.image_path, case["hashkey"]+"_first_frame.jpg")),
                "Describe what’s on the table. Don’t mention the robot arm."
            )
            print(response)
            result.append(dict(
                key = case["hashkey"],
                response = response,
                gt = case["annotation"]["Caption"]
            ))
            count+=1
        json.dump(result, open(f'{self.work_dir}/cap.json','w'),indent=4)

    def evaluate_QA(self):
        
        result = []
        count = 1 
        for case in tqdm(self.json):
            response = self.generate_one_sample(
                Image.open(osp.join(self.image_path, case["hashkey"]+"_first_frame.jpg")),
                case["annotation"]["QA"][0]["question"]
            )
            print(response)
            result.append(dict(
                key = case["hashkey"],
                response = response,
                gt = case["annotation"]["QA"][0]["answer"]
            ))
            count+=1
        json.dump(result, open(f'{self.work_dir}/qa.json','w'),indent=4)

    def evaluate_instruction(self):
        
        result = []
        count = 1 
        for case in tqdm(self.json):
            response = self.generate_one_sample(
                Image.open(osp.join(self.image_path, case["hashkey"]+"_first_frame.jpg")),
                case["annotation"]["CC"][0]["question"]
            )
            print(response)
            result.append(dict(
                key = case["hashkey"],
                response = response,
                gt = case["annotation"]["CC"][0]["answer"]
            ))
            count+=1
        json.dump(result, open(f'{self.work_dir}/instruct.json','w'),indent=4)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate InstructVLA on embodied language tasks.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/release_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction--image_aug/checkpoints/step-013500-epoch-01-loss=0.1093.pt",
        help="Path to the model checkpoint."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="outputs/release_ckpts/instructvla_finetune_v2_xlora_freeze_head_instruction--image_aug/vlmeval",
        help="Directory to save evaluation results."
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["caption", "qa", "instruction", "all"],
        default="all",
        help="Which evaluation task to run."
    )
    args = parser.parse_args()

    model = InstructVLA(model_path=args.model_path, work_dir=args.work_dir)

    if args.task in ["caption", "all"]:
        print('Evaluating captioning...')
        model.evaluate_captioning()
    if args.task in ["qa", "all"]:
        print('Evaluating question-answering...')
        model.evaluate_QA()
    if args.task in ["instruction", "all"]:
        print('Evaluating instruction following...')
        model.evaluate_instruction()