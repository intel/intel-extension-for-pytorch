import os
import time
import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from typing import Optional, Dict, Sequence
import io
#import utils
import copy
import pickle

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Llama-70B-Dataset")

import random

data_batch=64

class Dataset():
    def __init__(self, model_name=None, total_sample_count=24576, perf_count_override=None, dataset_path=None, device="cpu"):
        self.model_name = model_name or "mixtral/Mixtral-8x7B-Instruct-v0.1"
        self.dataset_path = dataset_path
        self.max_length = 1024
        self.device = device

        #self.total_sample_count = total_sample_count

        self.load_tokenizer()
        self.load_processed_dataset()

        self.total_sample_count = min(len(self.input_ids), total_sample_count)
        self.perf_count = perf_count_override or self.total_sample_count

    def load_tokenizer(self):
        """ Returns tokenizer """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,)

        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-70b-chat-hf",
            model_max_length=1024,
            padding_size="left",
            use_fast=True,)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llama_tokenizer.pad_token = self.llama_tokenizer.pad_token

    def load_processed_dataset(self):
        if not os.path.isfile(self.dataset_path):
            log.warn("Processed pickle file {} not found. Please check that the path is correct".format(self.dataset_path))

        print("Loading dataset...")
        import pandas as pd
        processed_data = pd.read_pickle(self.dataset_path)

        input_tokens = processed_data['tok_input']

        self.input_ids = []
        self.input_lens = []
        self.attention_masks = []

        ids_batch = []
        end = time.time()
        for i,ids in enumerate(input_tokens):
            ids_batch.append(ids)
            if len(ids_batch)==data_batch:
                end_int = time.time()
                input_strs = self.llama_tokenizer.batch_decode(ids_batch, skip_special_tokens=True)
                for input_str in input_strs:
                    ids = self.tokenizer.encode(input_str)
                    input_ids = torch.tensor(ids, dtype=torch.int32).view(1,-1).to(self.device)
                    attn_mask = torch.ones_like(input_ids)
                    self.input_ids.append(input_ids)
                    self.attention_masks.append(attn_mask)
                    self.input_lens.append(input_ids.shape[-1])
                ids_batch.clear()

        if len(ids_batch)>0:
            input_str = self.llama_tokenizer.batch_decode(ids_batch, skip_special_tokens=True)
            for input_str in input_strs:
                ids = self.tokenizer.encode(input_str)
                input_ids = torch.tensor(ids, dtype=torch.int32).view(1,-1).to(self.device)
                attn_mask = torch.ones_like(input_ids)
                self.input_ids.append(input_ids)
                self.attention_masks.append(attn_mask)
                self.input_lens.append(input_ids.shape[-1])
            del(ids_batch)
        print("Finished loading dataset.")

    def postProcess(self, out_tokens, input_seq_lens=None, query_id_list=None, sample_index_list=None):
        """ Postprocesses output prediction """

        #TODO: Create response object in postProcess(?)
        """
        preds = []
        for i in range(out_tokens.shape[0]):
            #pred = out_tokens[i].reshape(-1).cpu().numpy() # Slice up to original input length as below?

            input_len = input_seq_lens[i] if input_seq_lens else 0
            pred = out_tokens[i, input_len:].reshape(-1).cpu().numpy()
            preds.append(pred)
        """
        # Everything is padded to max_len (1024), so prune the input and parse to numpy
        output_seq = out_tokens[:, 1024:].cpu().numpy()
        assert len(query_id_list) == output_seq.shape[0]

        # Save outputs
        if not os.path.exists("run_outputs"):
            os.makedirs("run_outputs")
        fname = "q" + "_".join([str(i) for i in query_id_list])
        fname = f"run_outputs/{fname}.pkl"
        with open(fname, mode='wb') as f:
            d = {"query_ids": query_id_list,
                 "outputs": output_seq}
            print(f"Saving outputs to {fname}")
            pickle.dump(d, f)

        return output_seq

    def LoadSamplesToRam(self, sample_list):
        pass

    def UnloadSamplesFromRam(self, sample_list):
        pass

    def __del__(self):
        pass