from modules.utils import load_yaml, save_yaml, get_logger

from modules.earlystoppers import EarlyStopper
from modules.recorders import Recorder
from modules.datasets import QADataset
from modules.trainer import Trainer

from modules.optimizers import get_optimizer
from modules.metrics import get_metric
from modules.losses import get_loss
from models.utils import get_model

from transformers import ElectraTokenizerFast

from torch.utils.data import DataLoader
import torch

from datetime import datetime, timezone, timedelta
import numpy as np
import random
import os
import copy

import wandb

# Root directory
PROJECT_DIR = os.path.dirname(__file__)

# Load config
config_path = os.path.join(PROJECT_DIR, 'config', 'train_config.yml')
config = load_yaml(config_path)

# Train Serial
kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

# Recorder directory
DEBUG = config['TRAINER']['debug']
print(f'debug {DEBUG}')

RECORDER_DIR = os.path.join(PROJECT_DIR, 'results', 'train', train_serial)
os.makedirs(RECORDER_DIR, exist_ok=True)

# Data directory
DATA_DIR = config['DIRECTORY']['dataset']

# Seed
torch.manual_seed(config['TRAINER']['seed'])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(config['TRAINER']['seed'])
random.seed(config['TRAINER']['seed'])

# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['TRAINER']['gpu'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__': 
    """
    00. Set Logger
    """
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")
    print('/'.join(logger.handlers[0].baseFilename.split('/')[:-1]))

    """
    01. Load data
    """
    # Load tokenizer

    tokenizer_dict = {'ElectraTokenizerFast': ElectraTokenizerFast}

    tokenizer = tokenizer_dict[config['TRAINER']['tokenizer']].from_pretrained(config['TRAINER']['pretrained'])


    # Load Dataset
    if not DEBUG and os.path.isfile(os.path.join(DATA_DIR,'train_dataset.pt')):
        train_dataset = torch.load(os.path.join(DATA_DIR,'train_dataset.pt'))
        val_dataset = torch.load(os.path.join(DATA_DIR,'val_dataset.pt'))
        logger.info("loaded existing .pt")
    else:
        train_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'train.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'train')
        val_dataset = QADataset(data_dir=os.path.join(DATA_DIR, 'train.json'), tokenizer = tokenizer, max_seq_len = 512, mode = 'val')
        if not DEBUG:
            torch.save(train_dataset, os.path.join(DATA_DIR,'train_dataset.pt'))
            torch.save(val_dataset, os.path.join(DATA_DIR,'val_dataset.pt'))
        logger.info("loaded data, created .pt")


    
    if DEBUG:
        print(len(train_dataset), len(val_dataset))
        for i in range(10):
            txt = train_dataset[i]['input_ids']
            start_idx = train_dataset[i]['start_positions']
            end_idx = train_dataset[i]['end_positions']
            print(tokenizer.decode(txt[start_idx:end_idx]))
    
    # DataLoader
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['DATALOADER']['batch_size'],
                                  num_workers=config['DATALOADER']['num_workers'], 
                                  shuffle=config['DATALOADER']['shuffle'],
                                  pin_memory=config['DATALOADER']['pin_memory'],
                                  drop_last=config['DATALOADER']['drop_last'])
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=config['DATALOADER']['batch_size'],
                                num_workers=config['DATALOADER']['num_workers'], 
                                shuffle=False,
                                pin_memory=config['DATALOADER']['pin_memory'],
                                drop_last=config['DATALOADER']['drop_last'])

    logger.info(f"Load data, train:{len(train_dataset)} val:{len(val_dataset)}")

    """
    02. Set model
    """
    # Load model
    model_name = config['TRAINER']['model']
    model = get_model(model_name=model_name, pretrained=config['TRAINER']['pretrained']).to(device)

    """
    03. Set trainer
    """
    # Optimizer
    optimizer = get_optimizer(optimizer_name=config['TRAINER']['optimizer'])
    optimizer = optimizer(params=model.parameters(),lr=config['TRAINER']['learning_rate'])

    # Loss
    loss = get_loss(loss_name=config['TRAINER']['loss'])
    
    # Metric
    metrics = {metric_name: get_metric(metric_name) for metric_name in config['TRAINER']['metric']}
    
    # Early stoppper
    early_stopper = EarlyStopper(patience=config['TRAINER']['early_stopping_patience'],
                                mode=config['TRAINER']['early_stopping_mode'],
                                logger=logger)
    # AMP
    if config['TRAINER']['amp'] == True:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    # Trainer
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      device=device,
                      logger=logger,
                      tokenizer=tokenizer,
                      amp=amp if config['TRAINER']['amp'] else None,
                      interval=config['LOGGER']['logging_interval'])
    
    """
    Logger
    """
    # Recorder
    recorder = Recorder(record_dir=RECORDER_DIR,
                        model=model,
                        optimizer=optimizer,
                        scheduler=None,
                        amp=amp if config['TRAINER']['amp'] else None,
                        logger=logger)

    # !Wandb
    if config['LOGGER']['wandb'] == True:
        wandb_project_serial = 'template'
        wandb_username =  ''
        wandb.init(project=wandb_project_serial, dir=RECORDER_DIR, entity=wandb_username)
        wandb.run.name = train_serial
        wandb.config.update(config)
        wandb.watch(model)

    # Save train config
    save_yaml(os.path.join(RECORDER_DIR, 'train_config.yml'), config)

    """
    04. TRAIN
    """
    # Train
    n_epochs = config['TRAINER']['n_epochs']
    for epoch_index in range(n_epochs):

        # Set Recorder row
        row_dict = dict()
        row_dict['epoch_index'] = epoch_index
        row_dict['train_serial'] = train_serial
        
        """
        Train
        """
        print(f"Train {epoch_index}/{n_epochs}")
        logger.info(f"--Train {epoch_index}/{n_epochs}")
        trainer.train(dataloader=train_dataloader, epoch_index=epoch_index, tokenizer=tokenizer, mode='train')
        
        row_dict['train_loss'] = trainer.loss_mean
        row_dict['train_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict[f"train_{metric_str}"] = score
        trainer.clear_history()
        
        """
        Validation
        """
        print(f"Val {epoch_index}/{n_epochs}")
        logger.info(f"--Val {epoch_index}/{n_epochs}")
        trainer.train(dataloader=val_dataloader, epoch_index=epoch_index, tokenizer=tokenizer, mode='val')
        
        row_dict['val_loss'] = trainer.loss_mean
        row_dict['val_elapsed_time'] = trainer.elapsed_time 
        
        for metric_str, score in trainer.score_dict.items():
            row_dict[f"val_{metric_str}"] = score
        trainer.clear_history()
        
        """
        Record
        """
        recorder.add_row(row_dict)
        recorder.save_plot(config['LOGGER']['plot'])

        #!WANDB
        if config['LOGGER']['wandb'] == True:
            wandb.log(row_dict)

        """
        Early stopper
        """
        early_stopping_target = config['TRAINER']['early_stopping_target']
        early_stopper.check_early_stopping(loss=row_dict[early_stopping_target])

        if early_stopper.patience_counter == 0:
            recorder.save_weight(epoch=epoch_index)
            best_row_dict = copy.deepcopy(row_dict)
        
        if early_stopper.stop == True:
            logger.info(f"Early stopped, counter {early_stopper.patience_counter}/{config['TRAINER']['early_stopping_patience']}")
            
            if config['LOGGER']['wandb'] == True:
                wandb.log(best_row_dict)
            break
