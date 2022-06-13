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

from models.utils import get_model
from modules.datasets import QADataset
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

    test_dataset = QADataset(
        data_dir=os.path.join(DATA_DIR, "test.json"),
        tokenizer=tokenizer,
        max_seq_len=tokenizer.model_max_length,
        mode="test",
    )

    question_ids = test_dataset.question_ids

    BATCH_SIZE = train_config["DATALOADER"]["batch_size"]

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=train_config["DATALOADER"]["num_workers"],
        shuffle=False,
        pin_memory=train_config["DATALOADER"]["pin_memory"],
        drop_last=train_config["DATALOADER"]["drop_last"],
    )

    # Load model
    model_name = train_config["TRAINER"]["model"]
    # model_args = train_config['MODEL'][model_name]
    model = get_model(
        model_name=model_name, pretrained=train_config["TRAINER"]["pretrained"]
    ).to(device)

    checkpoint = torch.load(os.path.join(RECORDER_DIR, "model.pt"))

    if train_config["TRAINER"]["amp"] == True:
        from apex import amp

        model = amp.initialize(model, opt_level="O1")
        model.load_state_dict(checkpoint["model"])
        amp.load_state_dict(checkpoint["amp"])

    else:
        model.load_state_dict(checkpoint["model"])

    model.eval()
    pred_df = load_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    unk_token = tokenizer.unk_token
    sep_token = tokenizer.cls_token
    cls_token = tokenizer.sep_token
    with torch.set_grad_enabled(False):
        for batch_index, batch in enumerate(tqdm(test_dataloader, leave=True)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            context = batch["context"]
            q_ids = batch["question_id"]

            # Inference
            outputs = model(input_ids, attention_mask=attention_mask)

            start_score = outputs.start_logits
            end_score = outputs.end_logits

            start_idxes = torch.argmax(start_score, dim=1).cpu().tolist()
            end_idxes = torch.argmax(end_score, dim=1).cpu().tolist()

            y_pred = []
            for cont, txt, start_idx, end_idx in zip(
                context, input_ids, start_idxes, end_idxes
            ):
                if start_idx > end_idx:
                    ans_txt = ""

                pred_txt = txt[start_idx:end_idx]
                ans_txt = tokenizer.decode(pred_txt)
                if ans_txt == cls_token:
                    ans_txt == ""
                elif "#" in ans_txt:
                    ans_txt = ans_txt.replace("#", "")
                elif unk_token in ans_txt:
                    front_txt = tokenizer.decode(txt[:start_idx])
                    front_txt = front_txt.split(sep_token)[-1]
                    temp_cont = cont.strip()
                    for txts in front_txt.split(unk_token):
                        for txt in list(txts):
                            txt = txt.replace("#", "").strip()
                            if txt:
                                temp_cont_list = temp_cont.split(txt)
                                temp_cont = txt.join(temp_cont_list[1:])

                    ans_txt = ""
                    offset_mapping = tokenizer(
                        temp_cont, add_special_tokens=False, return_offsets_mapping=True
                    )["offset_mapping"][: len(pred_txt)]
                    for str_idx, end_idx in offset_mapping:
                        ans_txt += temp_cont[str_idx:end_idx]

                """
                ʻ미래의연구자ʼ는  /  미래의 연구자
                ʻ편리한세상ʼ에서는  /  편리한 세상
                ʻ위험인자평가서  /  위험인자 평가서
                ➄시료충진의  /  시료충진
                epost™  /  epost
                """
                ans_txt = re.sub("ʻ|ʼ|➄|™", "", ans_txt)

                y_pred.append(ans_txt)

            for q_id, pred in zip(q_ids, y_pred):
                pred_df.loc[pred_df["question_id"] == q_id, "answer_text"] = pred
    save_path = os.path.join(PREDICT_DIR, "prediction.csv")
    print(save_path)
    save_csv(save_path, pred_df)
