from modules.utils import load_yaml, save_pickle, save_json, load_csv, save_csv
from modules.datasets import QADataset
from models.utils import get_model

from transformers import ElectraTokenizerFast

from torch.utils.data import DataLoader

from datetime import datetime, timezone, timedelta
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import os
import torch

# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, 'config', 'predict_config.yml'))

# Serial
train_serial = predict_config['TRAIN']['train_serial']
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = train_serial + '_' + predict_timestamp

# Predict directory
PREDICT_DIR = os.path.join(PROJECT_DIR, 'results', 'predict', predict_serial)
os.makedirs(PREDICT_DIR, exist_ok=True)

# Data Directory
DATA_DIR = predict_config['DIRECTORY']['dataset']

# Train config
RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
train_config = load_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'))

# SEED
torch.manual_seed(predict_config['PREDICT']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config['PREDICT']['seed'])
random.seed(predict_config['PREDICT']['seed'])

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(predict_config['PREDICT']['gpu'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    # Load tokenizer

    tokenizer_dict = {'ElectraTokenizerFast': ElectraTokenizerFast}

    tokenizer = tokenizer_dict[train_config['TRAINER']['tokenizer']].from_pretrained(train_config['TRAINER']['pretrained'])
    
    # Load data
    
    test_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'test.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'test')
    
    question_ids = test_dataset.question_ids
    
    BATCH_SIZE = train_config['DATALOADER']['batch_size']
    
    test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=train_config['DATALOADER']['num_workers'], 
                                shuffle=False,
                                pin_memory=train_config['DATALOADER']['pin_memory'],
                                drop_last=train_config['DATALOADER']['drop_last'])

    # Load model
    model_name = train_config['TRAINER']['model']
    # model_args = train_config['MODEL'][model_name]
    model = get_model(model_name=model_name, pretrained=train_config['TRAINER']['pretrained']).to(device)

    checkpoint = torch.load(os.path.join(RECORDER_DIR, 'model.pt'))
    
    if train_config['TRAINER']['amp'] == True:
        from apex import amp
        model = amp.initialize(model, opt_level='O1')
        model.load_state_dict(checkpoint['model'])
        amp.load_state_dict(checkpoint['amp'])
        
    else:
        model.load_state_dict(checkpoint['model'])

    model.eval()
    pred_df = load_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

    for batch_index, batch in enumerate(tqdm(test_dataloader, leave=True)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Inference
        outputs = model(input_ids, attention_mask=attention_mask)
        
        start_score = outputs.start_logits
        end_score = outputs.end_logits
        
        start_idx = torch.argmax(start_score, dim=1).cpu().tolist()
        end_idx = torch.argmax(end_score, dim=1).cpu().tolist()
        
        y_pred = []
        for i in range(len(input_ids)):
            if start_idx[i] > end_idx[i]:
                output = ''
            
            ans_txt = tokenizer.decode(input_ids[i][start_idx[i]:end_idx[i]]).replace('#','')
            
            if ans_txt == '[CLS]':
                ans_txt == ''
            
            y_pred.append(ans_txt)
        

        q_end_idx = BATCH_SIZE*batch_index + len(y_pred)
        for q_id, pred in zip(question_ids[BATCH_SIZE*batch_index:q_end_idx], y_pred):
            pred_df.loc[pred_df['question_id'] == q_id,'answer_text'] = pred
            

    save_csv(os.path.join(PREDICT_DIR, 'prediction.csv'), pred_df)