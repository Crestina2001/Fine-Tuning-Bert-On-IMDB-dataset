import transformers
from utils import count_label_ratio, compute_acc, set_params, plot_losses
import torch
from models import BERT_Arch
from transformers import AdamW, AutoModel
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn as nn
from train import train_epoch, evaluate
from preprocess import random_sample, get_dataloader
import logging
import json
import os

def train_one_config(data_packed, device, params):
    # set the best valid loss during training
    best_valid_loss = float('inf')
    # unpack the data
    train_data, val_data, test_data, train_labels = data_packed
    # unpack the hyperparameters
    learning_rate, batch_size, num_epoch, dropout , frozen= \
        params['lr'], params['bs'], params['epochs'], params['dropout'], params['frozen']
    print(f'learning_rate:{learning_rate}, batch_size:{batch_size}, num_epoch:{num_epoch}, dropout:{dropout}, frozen:{frozen}')
    logging.info(f'learning_rate:{learning_rate}, batch_size:{batch_size}, num_epoch:{num_epoch}, dropout:{dropout}, frozen:{frozen}')
    # get the data loader
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(train_data, val_data, test_data, batch_size)
    '''
    Loss function
    '''  
    #compute the class weights
    class_wts = compute_class_weight(class_weight='balanced', classes =np.unique(train_labels), y=train_labels)
    # convert class weights to tensor
    weights= torch.tensor(class_wts,dtype=torch.float)
    weights = weights.to(device)
    # loss function
    cross_entropy  = nn.NLLLoss(weight=weights) 
    '''
    Set the model
    '''
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained("bert-base-uncased")

    # Freeze the first `frozen` layers
    for name, param in bert.named_parameters():
        if name.startswith('embeddings') and frozen['embeddings']:
            param.requires_grad = False
        if name.startswith('encoder.layer'):
            layer_num = int(name.split('.')[2])
            if layer_num < frozen['encoder']:
                param.requires_grad = False
        if name.startswith('pooler') and frozen['pooler']:
            param.requires_grad = False

    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(bert, dropout)

    model= nn.DataParallel(model)
    model = model.to(device)   
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = learning_rate)
    train_losses = []
    valid_losses = []
    for epoch in range(num_epoch):
        print('\n Epoch {:} / {:}'.format(epoch + 1, num_epoch))
        logging.info('Epoch {:} / {:}'.format(epoch + 1, num_epoch))
        
        #train model
        train_loss, _ = train_epoch(train_dataloader, device, model, optimizer, cross_entropy)
        
        #evaluate model
        valid_loss, _ = evaluate(val_dataloader, model, cross_entropy, device)
        
        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("SAVING MODEL")
            logging.info("SAVING MODEL")
            torch.save(model.state_dict(), 'saved_weights.pt')
        
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        logging.info(f'Training Loss: {train_loss:.3f}')
        logging.info(f'Validation Loss: {valid_loss:.3f}')
    plot_losses(train_losses, valid_losses, params)
    # load weights of best model
    path = 'saved_weights.pt'
    model.load_state_dict(torch.load(path))
    acc=compute_acc(model,test_dataloader, device)
    print(f'accuracy:{acc}')
    logging.info(f'accuracy:{acc}')
    return best_valid_loss, acc
    

def main_func():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    '''
    Get dataloaders and labels
    '''
    train_data, val_data, test_data, train_labels = random_sample('IMDB_Dataset.csv')
    # pack the data
    data_packed = train_data, val_data, test_data, train_labels

    running_records = []
    for frozen_encoder in [0]:
        for frozen_embedding in [False]:
            for frozen_pooler in [True, False]:
                best_valid_loss = float('inf')
                frozen = {
                        'encoder': frozen_encoder,
                        'embeddings': frozen_embedding,
                        'pooler': frozen_pooler
                    }
                for i in range(6):
                    params=set_params(frozen=frozen)
                    val_loss, acc = train_one_config(data_packed, device, params)
                    if best_valid_loss > val_loss:
                        record = params.copy()
                        record['val_loss'], record['acc'] = val_loss, acc
                running_records.append(record)

        
    '''
    for i in range(32):
        params=set_params()
        record = params.copy()
        record['val_loss'], record['acc'] = train_one_config(data_packed, device, params)
        running_records.append(record)
    '''
    '''
    for frozen_encoder in [0, 4, 8, 12]:
        for frozen_embedding in [True, False]:
            for frozen_pooler in [True, False]:
                frozen = {
                    'encoder': frozen_encoder,
                    'embeddings': frozen_embedding,
                    'pooler': frozen_pooler
                    }
                params=set_params(frozen=frozen, epochs=4, bs=32, lr = 2e-5, dropout=0)
                record = params.copy()
                record['val_loss'], record['acc'] = train_one_config(data_packed, device, params)
                running_records.append(record)
    '''
    print(running_records)
    logging.info(running_records)
    # Create the directory if it does not exist
    file_name = "grid_search_result.json"

    # Dump the results into the file
    with open(file_name, 'w') as file:
        json.dump(running_records, file)


if __name__=="__main__":
    # Set up logging configuration
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    main_func()
        