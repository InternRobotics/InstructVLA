import argparse
import json
from collections import defaultdict
import re
import os
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm

from copy import deepcopy
from collections import OrderedDict

import re
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

from scipy.spatial.distance import cosine
from glob import glob
import pickle
import copy

from sentence_transformers import SentenceTransformer, util
from transformers import AutoModel, AutoTokenizer

class Evaluator():
    def __init__(self,directory_path,eval_bs) -> None:
        self.eval_bs = eval_bs
        self.directory_path = directory_path
        self.simcse_tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
        self.simcse_model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large").to("cuda")

        self.sbert_model = SentenceTransformer('all-mpnet-base-v2',device="cuda")

    @staticmethod
    def to_coco(kvs, keys):
        res = defaultdict(list)
        for k in keys:
            if k in kvs:
                caps = kvs[k]
                for c in caps:
                    res[k].append({'caption': c})
            else:
                res[k].append({'caption': ''})
        return res

    def evaluate(self,ground_truths,prediction,verbose = True):

        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
        tokenizer = PTBTokenizer()
        ref_sent = ground_truths
        hypo_sent = prediction
        final_scores = {}
        ref_coco = tokenizer.tokenize(self.to_coco(ref_sent, ref_sent.keys()))
        hypo_coco = tokenizer.tokenize(self.to_coco(hypo_sent, ref_sent.keys()))
        for scorer, method in scorers:
            if verbose:
                print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(ref_coco, hypo_coco)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    @staticmethod
    def clean_answer(data):
        """
        LEO clean strategy
        """
        data = data.lower()
        data = re.sub('[ ]+$' ,'', data)
        data = re.sub('^[ ]+' ,'', data)
        data = re.sub(' {2,}', ' ', data)

        data = re.sub('\.[ ]{2,}', '. ', data)
        data = re.sub('[^a-zA-Z0-9,\'\s\-:]+', '', data)
        data = re.sub('ç' ,'c', data)
        data = re.sub('’' ,'\'', data)
        data = re.sub(r'\bletf\b' ,'left', data)
        data = re.sub(r'\blet\b' ,'left', data)
        data = re.sub(r'\btehre\b' ,'there', data)
        data = re.sub(r'\brigth\b' ,'right', data)
        data = re.sub(r'\brght\b' ,'right', data)
        data = re.sub(r'\bbehine\b', 'behind', data)
        data = re.sub(r'\btv\b' ,'TV', data)
        data = re.sub(r'\bchai\b' ,'chair', data)
        data = re.sub(r'\bwasing\b' ,'washing', data)
        data = re.sub(r'\bwaslked\b' ,'walked', data)
        data = re.sub(r'\boclock\b' ,'o\'clock', data)
        data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

        # digit to word, only for answer
        data = re.sub(r'\b0\b', 'zero', data)
        data = re.sub(r'\bnone\b', 'zero', data)
        data = re.sub(r'\b1\b', 'one', data)
        data = re.sub(r'\b2\b', 'two', data)
        data = re.sub(r'\b3\b', 'three', data)
        data = re.sub(r'\b4\b', 'four', data)
        data = re.sub(r'\b5\b', 'five', data)
        data = re.sub(r'\b6\b', 'six', data)
        data = re.sub(r'\b7\b', 'seven', data)
        data = re.sub(r'\b8\b', 'eight', data)
        data = re.sub(r'\b9\b', 'nine', data)
        data = re.sub(r'\b10\b', 'ten', data)
        data = re.sub(r'\b11\b', 'eleven', data)
        data = re.sub(r'\b12\b', 'twelve', data)
        data = re.sub(r'\b13\b', 'thirteen', data)
        data = re.sub(r'\b14\b', 'fourteen', data)
        data = re.sub(r'\b15\b', 'fifteen', data)
        data = re.sub(r'\b16\b', 'sixteen', data)
        data = re.sub(r'\b17\b', 'seventeen', data)
        data = re.sub(r'\b18\b', 'eighteen', data)
        data = re.sub(r'\b19\b', 'nineteen', data)
        data = re.sub(r'\b20\b', 'twenty', data)
        data = re.sub(r'\b23\b', 'twenty-three', data)

        # misc
        # no1, mat2, etc
        data = re.sub(r'\b([a-zA-Z]+)([0-9])\b' ,r'\g<1>', data)
        data = re.sub(r'\ba\b ([a-zA-Z]+)' ,r'\g<1>', data)
        data = re.sub(r'\ban\b ([a-zA-Z]+)' ,r'\g<1>', data)
        data = re.sub(r'\bthe\b ([a-zA-Z]+)' ,r'\g<1>', data)

        data = re.sub(r'\bbackwards\b', 'backward', data)

        return data
    
    def special_token_filter(self,lan,clean = True,truncation = True,max_length = 256):
        """
        Usage:
            clean the language, remove stop words and special tokens
        Args:
            lan: List[str], language to be cleaned
            clean: bool, if apply LEO clean strategy
            truncation: to avoid crash pycocoevalcap the input sentence will be truncated to max_length
            max_length: You may set this to the max length of possible gt answer
        """
        replacements = {
        "ASSISTANT:": "",
        "ASSISTANT: ": "",
        "\n": "",
        "<s>": "",
        "</s>": "",
        "<unk>": "",
        "<p>": "",
        "</p>": "",
        "<ref>": "",
        "<|endoftext|>": ""  # for GPT2
        }
        for old, new in replacements.items():
            lan = lan.replace(old, new)
        lan = lan.strip()
        lan = re.sub(r'\s{2,}', ' ', lan)
        if truncation:
            if len(lan)>max_length:
                lan = lan[:max_length]
        if clean:
            lan = self.clean_answer(lan)
        return lan

    @staticmethod
    def refined_EM(data,gt,set_zero_as_error=True,not_refine=False):
        EM = []
        _data = copy.deepcopy(data)
        if not_refine:
            for ins in _data:
                    pred  = _data[ins][0]
                    if pred in gt[ins]:
                        EM.append(1)
                    else:
                        EM.append(0)
        else:
            for ins in _data:
                to_append = 0
                pred  = _data[ins][0]
                if set_zero_as_error:
                    if pred in [" ",""]:
                        pred = "@@@@@@@@-= Empty Answer =-@@@@@@@@@"
                for _gt in gt[ins]:
                    if pred == _gt:
                        to_append = 1
                        continue
                    elif "".join(pred.split()) in "".join(_gt.split()):
                        to_append = 1
                        continue
                    elif "".join(_gt.split()) in "".join(pred.split()):
                        to_append = 1
                        continue
                EM.append(to_append)
        return EM

    @staticmethod
    def print_formated_dict(lan):
        for key in lan:
            print(f"{key}:      {lan[key]}")

    def batch_eval(self,all_pred,all_gt,gt_count):
        """
        Args:
            gt_count(list): stores number of possible answers to a question
            all_pred(list): all prediction
            all_gt(list): all ground truth,   len(all_gt)>=len(all_pred)

        Return:
            tuple: all_sbert_sim,all_simcse_sim
        """
        len_of_pred = len(all_pred)
        with torch.no_grad():
            sbert_embeddings = self.sbert_model.encode(all_pred+all_gt,show_progress_bar=False,device="cuda")
            inputs = self.simcse_tokenizer(all_pred+all_gt, padding=True, truncation=True, return_tensors="pt").to("cuda")
            simcse_embeddings = self.simcse_model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        all_pred_sbert_embed = sbert_embeddings[:len_of_pred]
        all_pred_simcse_embed = simcse_embeddings[:len_of_pred]

        all_gt_sbert_embed = sbert_embeddings[len_of_pred:]
        all_gt_simcse_embed = simcse_embeddings[len_of_pred:]

        all_sbert_sim = []
        all_simcse_sim = []

        accumulated = 0
        for i in range(len(all_pred)):
            simcse_similarity = -100
            sbert_similarity = -100
            for j in range(accumulated,accumulated+gt_count[i]):
                sbert_similarity = max(sbert_similarity, util.cos_sim(all_pred_sbert_embed[i], 
                                                                        all_gt_sbert_embed[j])[0][0].item())
                simcse_similarity = max(simcse_similarity ,1 - cosine(all_pred_simcse_embed[i].cpu().detach().numpy(), 
                                                                        all_gt_simcse_embed[j].cpu().detach().numpy())) 
            all_sbert_sim.append(sbert_similarity)
            all_simcse_sim.append(simcse_similarity)
            accumulated+=gt_count[i]
        torch.cuda.empty_cache()
        return all_sbert_sim,all_simcse_sim


    def load_data_and_eval(self,max_length=1024):
        all_pred = []
        lan_gt = {}
        lan_pred = {}

        all_simcse_similarity = []
        all_sbert_similarity = []

        all_pred_files = [self.directory_path]
        for filename in all_pred_files:
            with open(filename, 'r') as file:
                all_pred+=json.load(file)
        bar = tqdm(all_pred)

        batch_lan_pred = []
        batch_lan_gt = []
        count_gt = []

        for idx,result in enumerate(bar):
            key = result['key']
            pred = self.special_token_filter(result["response"],clean=True,truncation=True,max_length=max_length)
            lan_pred[key] = [pred]
            lan_gt[key] = [self.special_token_filter(result["gt"],clean=True,truncation=True,max_length=max_length)]
            batch_lan_pred += lan_pred[key]
            batch_lan_gt += lan_gt[key]
            count_gt += [len(lan_gt[key])]
            if idx % self.eval_bs==0:
                score = self.batch_eval(batch_lan_pred,batch_lan_gt,count_gt)
                all_simcse_similarity+=score[1]
                all_sbert_similarity+=score[0]

                batch_lan_pred = []
                batch_lan_gt = []
                count_gt = []
        if len(batch_lan_pred):
            score = self.batch_eval(batch_lan_pred,batch_lan_gt,count_gt)
            all_simcse_similarity+=score[1]
            all_sbert_similarity+=score[0]
        
        assert len(all_simcse_similarity) == len(all_pred)

        final_scores = self.evaluate(ground_truths=lan_gt,
                                    prediction=lan_pred)
        self.print_formated_dict(final_scores)

        EM = self.refined_EM(lan_pred,lan_gt,not_refine=True)
        print(f"EM:         { sum(EM)/len(EM)}")
        EM_refine = self.refined_EM(lan_pred,lan_gt,False)
        print(f"refined EM: { sum(EM_refine)/len(EM_refine)}")

        print(f"simcse:     {sum(all_simcse_similarity)/len(all_simcse_similarity)}")
        print(f"sbert:      {sum(all_sbert_similarity)/len(all_sbert_similarity)}")
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--directory_path', type=str,default="[TBD]", help='path to json files')
    parser.add_argument('--eval_bs', type=int, default=500, help='evaluation batch size')

    args = parser.parse_args()
    directory_path = args.directory_path
    eval_bs = args.eval_bs
    
    print(f"evaluating files under {directory_path} ...")

    eval = Evaluator(
        directory_path=directory_path,
        eval_bs=eval_bs
    )
    eval.load_data_and_eval(max_length=1024)


        