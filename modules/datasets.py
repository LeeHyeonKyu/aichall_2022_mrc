"""Dataset

modified from : 
https://huggingface.co/transformers/v3.3.1/custom_datasets.html#question-answering-with-squad-2-0
https://towardsdatascience.com/how-to-fine-tune-a-q-a-transformer-86f91ec92997

"""

import os
import random
from collections import defaultdict
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from modules.utils import load_json, load_pickle


class CustomQADataset(Dataset):
    def __init__(
        self, dataset, tokenizer, max_seq_len, mode, question_shuffle_aug, pororo_aug, gpt_aug, debug=False
    ):
        self.dataset = dataset.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.question_shuffle_aug = question_shuffle_aug
        self.pororo_aug = pororo_aug
        self.gpt_aug = gpt_aug
        self.debug = debug

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        encoded_sample, context, answers = self.preprocess(idx)
        encoded_sample = {k:torch.tensor(v) for k, v in encoded_sample.items()}
        encoded_sample['context'] = context
        encoded_sample["question_id"] = self.dataset.loc[idx, 'question_id']
        encoded_sample['answer_text'] = answers['text']
        return encoded_sample

    def preprocess(self, idx):
        if self.mode=='train' and random.random() > 0.2 and (self.question_shuffle_aug or self.pororo_aug or self.gpt_aug):
            question, context, answers = self.augmentation(idx)
        else:
            question, context, answers = self.dataset.loc[idx, 'question'], self.dataset.loc[idx, 'context'], self.dataset.loc[idx, 'answers']
        
        encoded_sample = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
            padding="max_length",
        )

        encoded_sample = self.add_token_positions(encoded_sample, answers)
        return encoded_sample, context, answers

    def augmentation(self, idx):
        sr_sample = self.dataset.loc[idx]
        key='question'
        if sr_sample['is_impossible']:
            if self.pororo_aug and self.gpt_aug:
                # key = random.choice(['pororo_paraphrase_question', 'gpt_paraphrase_question'])
                key = 'pororo_paraphrase_question'
            elif self.pororo_aug:
                key = 'pororo_paraphrase_question'
            elif self.gpt_aug:
                key = 'gpt_paraphrase_question'
            question = random.choice(sr_sample[key]) if key == 'pororo_paraphrase_question' else sr_sample[key]
            context = sr_sample['context']
            answers = sr_sample['answers']
        else:
            if self.pororo_aug and self.gpt_aug:
                key = random.choice(['pororo_paraphrase_question', 'gpt_paraphrase_question'])
            elif self.pororo_aug:
                key = 'pororo_paraphrase_question'
            elif self.gpt_aug:
                key = 'gpt_paraphrase_question'
            question = random.choice(sr_sample[key]) if key == 'pororo_paraphrase_question' else sr_sample[key]
            context = sr_sample['context']
            answers = sr_sample['answers']

            if self.question_shuffle_aug and random.random() > 0.5:
                content_id = sr_sample['content_id']
                paragraph_id = sr_sample['paragraph_id']
                tmp = self.dataset[self.dataset['content_id'] == content_id]
                tmp = tmp[tmp['is_impossible'] == False]
                tmp = tmp[tmp['paragraph_id'] != paragraph_id]
                if len(tmp['context'].values) > 0:
                    context = random.choice(tmp['context'].values)
                    answers = {'answer_end': 0, 'answer_start': 0, 'text': ""}

        return question, context, answers

    def add_token_positions(self, encoded_sample, answers):
        offsets = encoded_sample["offset_mapping"]
        input_ids = encoded_sample["input_ids"]
        sequence_ids = encoded_sample.sequence_ids(0)

        if answers["text"] == "":
            token_start_index = 0
            token_end_index = 0
        else:
            start_char = answers["answer_start"]
            end_char = answers["answer_end"]

            token_start_index = 0
            while sequence_ids[token_start_index] != (1):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1):
                token_end_index -= 1

            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1
            token_start_index -= 1

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            while token_end_index < len(input_ids) and offsets[token_end_index][0] < end_char:
                token_end_index += 1

        encoded_sample['start_positions'] = token_start_index
        encoded_sample['end_positions'] = token_end_index
        return encoded_sample


class QADataset(Dataset):
    def __init__(
        self, data_dir: str, tokenizer, max_seq_len: int, mode="train", debug=False, aug=False,
    ):
        self.mode = mode
        self.data = load_json(data_dir)

        # self.encodings = encodings
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.debug = debug
        self.aug = aug
        if mode == "test":
            self.encodings, self.question_ids, self.contexts = self.preprocess()
        else:
            self.encodings, self.answers, self.contexts = self.preprocess()

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, index: int):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["context"] = self.contexts[index]
        if self.mode == "test":
            item["question_id"] = self.question_ids[index]
        else:
            item["answer"] = self.answers[index]["text"]
        return item

    def preprocess(self):
        contexts, questions, answers, question_ids = self.read_squad()
        if self.mode == "test":
            encodings = self.tokenizer(
                questions,
                contexts,
                truncation="only_second",
                max_length=self.max_seq_len,
                return_offsets_mapping=True,
                padding="max_length",
            )
            return encodings, question_ids, contexts
        else:  # train or val
            self.add_end_idx(answers, contexts)
            encodings = self.tokenizer(
                questions,
                contexts,
                truncation="only_second",
                max_length=self.max_seq_len,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )
            self.add_token_positions(encodings, answers)

            return encodings, answers, contexts
          
    def question_shuffle_augmentation(self, dataset):
        for group in dataset['data']:
            aug_questions = []
            content_id = group['content_id']
            for passage in group['paragraphs']:
                for qa in passage['qas']:
                    if not qa['is_impossible']:
                        aug_qa = deepcopy(qa)
                        aug_qa['answers'] = []
                        aug_qa['is_impossible'] = True
                        aug_questions.append(aug_qa)
            
            for passage in group['paragraphs']:
                first_qa = passage['qas'][0]
                if not first_qa['is_impossible']:
                    random_aug_qa = random.sample(aug_questions, k=min(2, len(aug_questions)))
                    for aug_qa in random_aug_qa:
                        if aug_qa['question_id'] != first_qa['question_id']:
                            passage['qas'].append(aug_qa)
        return dataset

    def read_squad(self):
        contexts = []
        questions = []
        question_ids = []
        answers = []

        # train - val split
        if self.mode == "train":
            if self.aug:
                self.data = self.question_shuffle_augmentation(self.data)
            self.data["data"] = self.data["data"][
                : -1 * int(len(self.data["data"]) * 0.1)
            ]
        elif self.mode == "val":
            self.data["data"] = self.data["data"][
                -1 * int(len(self.data["data"]) * 0.1) :
            ]

        till = 100 if self.debug else len(self.data["data"])

        for group in tqdm(self.data["data"][:till]):
            for passage in group["paragraphs"]:
                context = passage["context"]
                for qa in passage["qas"]:
                    question = qa["question"]
                    if self.mode == "test":
                        contexts.append(context)
                        questions.append(question)
                        question_ids.append(qa["question_id"])
                    else:  # train or val
                        contexts.append(context)
                        questions.append(question)

                        if qa["is_impossible"]:
                            answers.append({"text": "", "answer_start": -1})
                        else:
                            answers.append(qa["answers"][0])

        # return formatted data lists
        return contexts, questions, answers, question_ids

    def add_end_idx(self, answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer["text"]
            start_idx = answer["answer_start"]
            end_idx = start_idx + len(gold_text)

            # in case the indices are off 1-2 idxs
            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            else:
                for n in [1, 2]:
                    if context[start_idx - n : end_idx - n] == gold_text:
                        answer["answer_start"] = start_idx - n
                        answer["answer_end"] = end_idx - n
                    elif context[start_idx + n : end_idx + n] == gold_text:
                        answer["answer_start"] = start_idx + n
                        answer["answer_end"] = end_idx + n

    def add_token_positions(self, encodings, answers):
        start_positions = []
        end_positions = []
        
        sample_mapping = encodings["overflow_to_sample_mapping"]
        offset_mapping = encodings["offset_mapping"]

        for i, offsets in enumerate(offset_mapping):
            input_ids = encodings["input_ids"][i]
            sequence_ids = encodings.sequence_ids(i)
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sample_index = sample_mapping[i]
            sample_answer = answers[i]

            if sample_answer["answer_start"] == -1:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = sample_answer["answer_start"]
                end_char = sample_answer["answer_end"]

                token_start_index = 0
                while sequence_ids[token_start_index] != (1):
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1):
                    token_end_index -= 1

                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                start_positions.append(token_start_index - 1)

                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                while token_end_index < len(input_ids) and offsets[token_end_index][0] < end_char:
                    token_end_index += 1
                end_positions.append(token_end_index)

        encodings.update(
            {"start_positions": start_positions, "end_positions": end_positions}
        )

def json_to_df(data_dir, mode, pororo_dir, gpt_dir):
    df_dataset = pd.DataFrame(columns=['question_id', 'question', 'paragraph_id', 'context', 'answers', 'is_impossible', 'content_id', 'pororo_paraphrase_question', 'gpt_paraphrase_question'])
    js_dataset = load_json(data_dir)
    pororo_paraphrase_dataset = load_pickle(pororo_dir)
    gpt_paraphrase_dataset = load_pickle(gpt_dir)

    for group in tqdm(js_dataset['data']):
        content_id = group['content_id']
        for passage in group['paragraphs']:
            paragraph_id = passage['paragraph_id']
            context = passage['context']
            for qa in passage['qas']:
                question_id = qa['question_id']
                question = qa['question']
                is_impossible = qa['is_impossible'] if 'is_impossible' in qa.keys() else True
                if is_impossible:
                    answer_text = ''
                    answer_start = 0
                    answer_end = 0
                else:
                    answer_text = qa['answers'][0]['text']
                    answer_start = qa['answers'][0]['answer_start']
                    answer_end = answer_start + len(answer_text)
                    for n in [0, 1, 2]:
                        if context[answer_start - n : answer_end - n] == answer_text:
                            answer_start -= n
                            answer_end -= n
                            break
                        elif context[answer_start + n : answer_end + n] == answer_text:
                            answer_start += n
                            answer_end += n
                            break

                tmp = pd.DataFrame(
                    data = {
                        'question_id':[question_id], 
                        'question':[question], 
                        'paragraph_id':[paragraph_id], 
                        'context':[context], 
                        'answers':[{'text':answer_text, 'answer_start':answer_start, 'answer_end':answer_end}],
                        'is_impossible':[is_impossible], 
                        'content_id':[content_id],
                        'pororo_paraphrase_question':[pororo_paraphrase_dataset[question]],
                        'gpt_paraphrase_question':[gpt_paraphrase_dataset[question] if not is_impossible else question]
                    })
                df_dataset = pd.concat([df_dataset, tmp], axis=0, ignore_index=True)

    return df_dataset

if __name__ == "__main__":
    pass
