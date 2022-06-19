import os
import random
import pickle
from time import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
# from apex import amp

# from modules.losses import ce_loss, joint_loss
from modules.utils import load_json

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        metrics,
        device,
        logger,
        amp,
        tokenizer,
        interval=100,
        grad_accum=1
    ):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device
        self.logger = logger
        self.amp = amp
        self.interval = interval
        self.tokenizer = tokenizer
        self.grad_accum = grad_accum

        # History
        self.loss_sum = 0  # Epoch loss sum
        self.loss_mean = 0  # Epoch loss mean
        self.q_ids = list()
        self.y = list()
        self.y_preds = list()
        self.score_dict = dict()  # metric score
        self.elapsed_time = 0

    def train(self, mode, dataloader, tokenizer, random_masking, epoch_index=0):
        with torch.set_grad_enabled(mode == "train"):
            start_timestamp = time()
            self.model.train() if mode == "train" else self.model.eval()

            for batch_index, batch in enumerate(tqdm(dataloader, leave=True)):

                if mode == 'train' and random_masking:
                    batch["input_ids"] = self.ramdom_masking(batch["input_ids"])

                # initialize calculated gradients (from prev step)
                # self.optimizer.zero_grad()
                # pull all the tensor batches required for training
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device) if "token_type_ids" in batch.keys() else None

                # train model on batch and return outputs (incl. loss)
                # Inference
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    token_type_ids=token_type_ids
                )

                loss = self.loss_fn(start_positions, end_positions, outputs.start_logits, outputs.end_logits)
                # start_score = outputs.start_logits
                # end_score = outputs.end_logits

                start_idxes = torch.argmax(outputs.start_logits, dim=1).cpu().tolist()
                end_idxes = torch.argmax(outputs.end_logits, dim=1).cpu().tolist()

                # Update
                if mode == "train" and batch_index % self.grad_accum == 0:

                    if self.amp is None:
                        loss.backward()

                    else:
                        with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                elif mode in ["val", "test"]:
                    pass

                # History
                # self.filenames += filename
                self.loss_sum += loss.item()

                # create answer; list of strings
                for context, offsets, start_idx, end_idx, q_id, ans_txt in zip(batch["context"], batch["offset_mapping"], start_idxes, end_idxes, batch["question_id"], batch["answer_text"]):
                    if start_idx >= end_idx:
                        pred_txt = ""
                    else:
                        s = offsets[start_idx][0]
                        e = offsets[end_idx][0]
                        pred_txt = context[s:e]
                    self.y.append(ans_txt)
                    self.y_preds.append(pred_txt)
                    self.q_ids.append(q_id)
                    
                # for i in range(len(input_ids)):
                #     if start_idx[i] > end_idx[i]:
                #         output = ""

                #     self.y_preds.append(
                #         self.tokenizer.decode(input_ids[i][start_idx[i] : end_idx[i]])
                #     )
                #     self.y.append(
                #         self.tokenizer.decode(
                #             input_ids[i][start_positions[i] : end_positions[i]]
                #         )
                #     )

                # Logging
                if batch_index % self.interval == 0:
                    msg = f"batch: {batch_index}/{len(dataloader)} loss: {loss.item()}"
                    self.logger.info(msg)

            # Epoch history
            self.loss_mean = self.loss_sum / len(dataloader)  # Epoch loss mean

            # Metric

            for metric_name, metric_func in self.metrics.items():
                score = metric_func(self.y, self.y_preds)
                self.score_dict[metric_name] = score

            # Elapsed time
            end_timestamp = time()
            self.elapsed_time = end_timestamp - start_timestamp

    def clear_history(self):
        self.loss_sum = 0
        self.loss_mean = 0
        self.q_ids = list()
        self.y_preds = list()
        self.y = list()
        self.score_dict = dict()
        self.elapsed_time = 0

    def ramdom_masking(self, input_ids, threshold = 0.05):
        for sample_idx, sample_input_id in enumerate(input_ids):
            for token_idx, token in enumerate(sample_input_id):
                if token == self.tokenizer.sep_token_id or token == self.tokenizer.eos_token_id:
                    break
                if token != self.tokenizer.cls_token_id and token != self.tokenizer.bos_token_id and random.random() < threshold:
                    input_ids[sample_idx][token_idx] = self.tokenizer.mask_token_id
        return input_ids

def apply_train_distribution(start_score, end_score, diff_dict, n_best=5, smooth=10, use_fn=True):
    if not use_fn:
        start_idxes = torch.argmax(start_score, dim=1).tolist()
        end_idxes = torch.argmax(end_score, dim=1).tolist()
    else:
        start_topk = torch.topk(start_score, n_best, axis=1)
        end_topk = torch.topk(end_score, n_best, axis=1)

        start_topk_val = start_topk.values.repeat_interleave(n_best, 1)
        end_topk_val = end_topk.values.repeat(1, n_best)

        start_topk_idxes = start_topk.indices.repeat_interleave(n_best, 1)
        end_topk_idxes = end_topk.indices.repeat(1, n_best)
        end_start_diff = end_topk_idxes - start_topk_idxes
        end_start_diff = end_start_diff.float()
        end_start_diff.apply_(lambda x: diff_dict[x]/smooth)
        
        tot_logit = start_topk_val + end_topk_val + end_start_diff
        tot_start_end_idx = torch.argmax(tot_logit, dim=1)

        start_idxes, end_idxes = [], []
        for start_idx, end_idx, start_end_idx in zip(start_topk_idxes.tolist(), end_topk_idxes.tolist(), tot_start_end_idx.tolist()):
            start_idxes.append(start_idx[start_end_idx])
            end_idxes.append(end_idx[start_end_idx])
    
    return start_idxes, end_idxes

def get_token_distance_distribution(tokenizer, train_path):
    train = load_json(train_path)

    tmp_dict = defaultdict(int)
    diff_dict = defaultdict(lambda: -100)
    num_of_train_sample = 0

    for group in train['data']:
        for passage in group['paragraphs']:
            for qa in passage['qas']:
                answers = qa['answers']
                if len(answers) > 0:
                    answer = qa['answers'][0]['text']
                    token_distance = len(tokenizer(answer, add_special_tokens=False)['input_ids'])
                else:
                    token_distance = 0
                num_of_train_sample += 1
                tmp_dict[token_distance] += 1

    for k, v in tmp_dict.items():
        diff_dict[k] = v/num_of_train_sample
    
    return diff_dict