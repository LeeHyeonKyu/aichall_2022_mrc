"""Dataset

modified from : 
https://huggingface.co/transformers/v3.3.1/custom_datasets.html#question-answering-with-squad-2-0
https://towardsdatascience.com/how-to-fine-tune-a-q-a-transformer-86f91ec92997

"""

from torch.utils.data import Dataset
from modules.utils import load_json
import numpy as np
import os
import torch


class QADataset(Dataset):
    
    def __init__ (self, data_dir: str, tokenizer, max_seq_len: int, mode = 'train', debug = False):
        self.mode = mode
        self.data = load_json(data_dir)
        
        # self.encodings = encodings
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.debug = debug
        if mode == 'test':
            self.encodings, self.question_ids = self.preprocess()
        else:
            self.encodings, self.answers = self.preprocess()
        
    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, index: int):
        return {key: torch.tensor(val[index]) for key, val in self.encodings.items()}

    
    def preprocess(self):
        contexts, questions, answers, question_ids = self.read_squad()
        if self.mode == 'test':
            encodings = self.tokenizer(contexts, questions, truncation=True, max_length = self.max_seq_len, padding=True)
            return encodings, question_ids
        else: # train or val
            self.add_end_idx(answers, contexts)
            encodings = self.tokenizer(contexts, questions, truncation=True, max_length = self.max_seq_len, padding=True)
            self.add_token_positions(encodings, answers)
        
            return encodings, answers
    
    def read_squad(self):
        contexts = []
        questions = []
        question_ids = []
        answers = []
        
        # train - val split
        if self.mode == 'train':
            self.data['data'] = self.data['data'][:-1*int(len(self.data['data'])*0.1)]
        elif self.mode == 'val':
            self.data['data'] = self.data['data'][-1*int(len(self.data['data'])*0.1):]
        
        
        till = 100 if self.debug else len(self.data['data'])
        

        for group in self.data['data'][:till]:
            for passage in group['paragraphs']:
                context = passage['context']
                for qa in passage['qas']:
                    question = qa['question']
                    if self.mode == 'test':
                        contexts.append(context)
                        questions.append(question)
                        question_ids.append(qa['question_id'])
                    else: # train or val
                        for ans in qa['answers']:
                            contexts.append(context)
                            questions.append(question)

                            if qa['is_impossible']:
                                answers.append({'text':'','answer_start':-1})
                            else:
                                answers.append(ans)
                
        # return formatted data lists
        return contexts, questions, answers, question_ids
    
    def add_end_idx(self, answers, contexts):
        for answer, context in zip(answers, contexts):
            gold_text = answer['text']
            start_idx = answer['answer_start']
            end_idx = start_idx + len(gold_text)

            # in case the indices are off 1-2 idxs
            if context[start_idx:end_idx] == gold_text:
                answer['answer_end'] = end_idx
            else:
                for n in [1, 2]:
                    if context[start_idx-n:end_idx-n] == gold_text:
                        answer['answer_start'] = start_idx - n
                        answer['answer_end'] = end_idx - n
                    elif context[start_idx+n:end_idx+n] == gold_text:
                        answer['answer_start'] = start_idx + n
                        answer['answer_end'] = end_idx + n

    def add_token_positions(self, encodings, answers):
        # should use Fast tokenizer
        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            if answers[i]['answer_start'] == -1:
                # set [CLS] token as answer if is_impossible
                start_positions.append(0)
                end_positions.append(1)
            else:
                start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))

                assert 'answer_end' in answers[i].keys(), f'no answer_end at {i}'
                end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

            # answer passage truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            # end position cannot be found, shift until found
            shift = 1
            while end_positions[-1] is None:
                end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
                shift += 1
        # char-based -> token based
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

    
if __name__ == '__main__':
    pass