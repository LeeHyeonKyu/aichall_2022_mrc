"""Dataset

modified from : 
https://huggingface.co/transformers/v3.3.1/custom_datasets.html#question-answering-with-squad-2-0
https://towardsdatascience.com/how-to-fine-tune-a-q-a-transformer-86f91ec92997

"""

import os
import random
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset

from modules.utils import load_json


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
            self.encodings, self.question_ids = self.preprocess()
        else:
            self.encodings, self.answers = self.preprocess()

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, index: int):
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}

    def preprocess(self):
        contexts, questions, answers, question_ids = self.read_squad()
        if self.mode == "test":
            encodings = self.tokenizer(
                questions,
                contexts,
                truncation="only_second",
                max_length=self.max_seq_len,
                padding="max_length",
            )
            return encodings, question_ids
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

            return encodings, answers

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
        
        sample_mapping = encodings.pop("overflow_to_sample_mapping")
        offset_mapping = encodings.pop("offset_mapping")

        for i, offsets in enumerate(offset_mapping):
            input_ids = encodings['input_ids'][i]
            sequence_ids = encodings.sequence_ids(i)
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            sample_index = sample_mapping[i]
            sample_answer = answers[i]
            
            if sample_answer['answer_start'] == -1:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = sample_answer['answer_start']
                end_char = sample_answer['answer_end']

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
                while offsets[token_end_index][0] < end_char:
                    token_end_index += 1
                end_positions.append(token_end_index)

        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


if __name__ == "__main__":
    pass
