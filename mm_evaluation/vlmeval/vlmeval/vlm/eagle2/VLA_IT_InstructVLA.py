import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor
import warnings
from PIL import Image


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
from .cogactvla_eagle_dual_sys_v2_meta_query_v2 import load, load_vla
from PIL import Image
from tqdm import tqdm
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class InstructVLA():

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/head_balation/sys12_meta_query_full_finetune_sync_cotraining_v2_xlora_freeze_head_instruction_long--image_augstage2/checkpoints/step-013500-epoch-01-loss=0.1093.pt', load_in_8bit=False, version='V2.0', **kwargs):
        assert model_path is not None

        self.json_path = "/mnt/petrelfs/yangshuai1/rep/cogact_with_history/data_pipeline/language_evaluation_with_gt.json"
        self.json = json.load(open(self.json_path))
        self.image_path = "/mnt/petrelfs/yangshuai1/rep/cogact_with_history/data_pipeline/language_evaluation"

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
        self.model = load_vla(self.model_path, hf_token='REMOVED_TOKEN', stage = "stage2").eval().to(torch.bfloat16)
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
            if count%100 == 0:
                del self.model
                self.load_model()
            count+=1
        json.dump(result, open('/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/vlmeval/vla_it/InstructVLA/cap.json','w'),indent=4)

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
            if count%100 == 0:
                del self.model
                self.load_model()
            count+=1
        json.dump(result, open('/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/vlmeval/vla_it/InstructVLA/qa.json','w'),indent=4)

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
            if count%100 == 0:
                del self.model
                self.load_model()
            count+=1
        json.dump(result, open('/mnt/petrelfs/yangshuai1/rep/cogact_with_history/outputs/vlmeval/vla_it/InstructVLA/instruct.json','w'),indent=4)


if __name__ == "__main__":
    model = InstructVLA()
    model.evaluate_captioning()
    # python -m vlmeval.vlm.eagle2.VLA_IT_InstructVLA
    # model.evaluate_QA()
    # model.evaluate_instruction()