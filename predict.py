import os
import re
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ElectraTokenizerFast
from collections import defaultdict

from models.utils import get_model
from modules.datasets import QADataset
from modules.trainer import apply_train_distribution, get_token_distance_distribution
from modules.utils import load_csv, load_yaml, save_csv, save_json, save_pickle
from modules.preprocessing import get_tokenizer

# Config
PROJECT_DIR = os.path.dirname(__file__)
predict_config = load_yaml(os.path.join(PROJECT_DIR, "config", "predict_config.yml"))

# Serial
train_serial = predict_config["TRAIN"]["train_serial"]
kst = timezone(timedelta(hours=9))
predict_timestamp = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")
predict_serial = train_serial + "_" + predict_timestamp

# Predict directory
PREDICT_DIR = os.path.join(PROJECT_DIR, "results", "predict", predict_serial)
os.makedirs(PREDICT_DIR, exist_ok=True)

# Data Directory
DATA_DIR = predict_config["DIRECTORY"]["dataset"]

# Train config
RECORDER_DIR = os.path.join(PROJECT_DIR, "results", "train", train_serial)
train_config = load_yaml(os.path.join(RECORDER_DIR, "train_config.yml"))

# SEED
torch.manual_seed(predict_config["PREDICT"]["seed"])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(predict_config["PREDICT"]["seed"])
random.seed(predict_config["PREDICT"]["seed"])

# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(predict_config["PREDICT"]["gpu"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Load tokenizer

    tokenizer = get_tokenizer(
        train_config["TRAINER"]["tokenizer"], train_config["TRAINER"]["pretrained"]
    )

    # Load data

    file_name = "test.json" if not train_config["TRAINER"]["debug"] else "sample.json"
    test_dataset = QADataset(
        data_dir=os.path.join(DATA_DIR, file_name),
        tokenizer=tokenizer,
        max_seq_len=tokenizer.model_max_length,
        mode="test",
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=train_config["DATALOADER"]["batch_size"],
        num_workers=train_config["DATALOADER"]["num_workers"],
        shuffle=False,
        pin_memory=train_config["DATALOADER"]["pin_memory"],
        drop_last=train_config["DATALOADER"]["drop_last"],
    )

    question_ids = test_dataset.question_ids
    pred_df = load_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # get train token distance distribution
    if predict_config["PREDICT"]["distribution_use"]:
        diff_dict = get_token_distance_distribution(tokenizer, os.path.join(DATA_DIR, "train.json"))
    else:
        diff_dict = {}

    """
    KFold Loop
    """
    all_fold_preds = defaultdict(list)
    n_fold = train_config["TRAINER"]["KFold"] if "KFold" in train_config["TRAINER"].keys() else 1
    for fold_idx in range(n_fold):
        fold_preds = defaultdict(list)

        # Load model
        model_name = train_config["TRAINER"]["model"]
        # model_args = train_config['MODEL'][model_name]
        model = get_model(
            model_name=model_name, pretrained=train_config["TRAINER"]["pretrained"]
        ).to(device)

        weight_path = f"model_{fold_idx}.pt"
        if not os.path.isfile(os.path.join(RECORDER_DIR, weight_path)):
            weight_path = "model.pt"
        checkpoint = torch.load(os.path.join(RECORDER_DIR, weight_path))

        if train_config["TRAINER"]["amp"] == True:
            from apex import amp

            model = amp.initialize(model, opt_level="O1")
            model.load_state_dict(checkpoint["model"])
            amp.load_state_dict(checkpoint["amp"])

        else:
            model.load_state_dict(checkpoint["model"])

        model.eval()

        with torch.set_grad_enabled(False):
            for batch_index, batch in enumerate(tqdm(test_dataloader, leave=True)):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device) if "token_type_ids" in batch.keys() else None
                contexts = batch["context"]
                q_ids = batch["question_id"]

                # Inference
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                start_score = outputs.start_logits.detach().cpu()
                end_score = outputs.end_logits.detach().cpu()
                fold_preds['start_score'].extend(start_score)
                fold_preds['end_score'].extend(end_score)

                if fold_idx == 0:
                    all_fold_preds["context"].extend(batch["context"])
                    all_fold_preds["question_id"].extend(batch["question_id"])
                    all_fold_preds["offset_mapping"].extend(batch["offset_mapping"])

        all_fold_preds['start_score'].append(torch.stack(fold_preds['start_score']))
        all_fold_preds['end_score'].append(torch.stack(fold_preds['end_score']))

    all_fold_preds['start_score'] = sum(all_fold_preds['start_score']) / len(all_fold_preds['start_score'])
    all_fold_preds['end_score'] = sum(all_fold_preds['end_score']) / len(all_fold_preds['end_score'])

    y_pred, q_ids = [], []
    for context, offsets, start_score, end_score, question_id in zip(all_fold_preds['context'], all_fold_preds['offset_mapping'], all_fold_preds['start_score'], all_fold_preds['end_score'], all_fold_preds['question_id']):
        start_idxes, end_idxes = apply_train_distribution(
            start_score = start_score.unsqueeze(0), 
            end_score = end_score.unsqueeze(0),
            diff_dict = diff_dict,
            n_best=predict_config["PREDICT"]["distribution_n_best"], 
            smooth=predict_config["PREDICT"]["distribution_smooth"], 
            use_fn=predict_config["PREDICT"]["distribution_use"],
        )
        start_idx, end_idx = start_idxes[0], end_idxes[0]
        if start_idx >= end_idx:
            ans_txt = ""
        else:
            s = offsets[start_idx][0]
            e = offsets[end_idx][0]
            ans_txt = context[s:e]
        ans_txt = re.sub("ʻ|ʼ|➄|™|『|』", "", ans_txt)
        y_pred.append(ans_txt)
        q_ids.append(question_id)

    if train_config["TRAINER"]["debug"]:
        pred_df = pd.DataFrame({"question_id":q_ids, "answer_text":y_pred})
    else:
        # pred_df["question_id"] = q_ids
        # pred_df["answer_text"] = y_pred
        for q_id, pred in zip(q_ids, y_pred):
            pred_df.loc[pred_df["question_id"] == q_id, "answer_text"] = pred
    save_path = os.path.join(PREDICT_DIR, "prediction.csv")
    print(save_path)
    save_csv(save_path, pred_df)