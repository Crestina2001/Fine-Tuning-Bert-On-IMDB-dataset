import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def random_sample(csv_file, pretrained_model="bert-base-uncased"):
    # Load dataset
    df = pd.read_csv(csv_file)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Split dataset into train, validation, and test sets
    df2 = df.sample(n=6000)
    train_text, temp_text, train_labels, temp_labels = train_test_split(
        df2['review'], df2['sentiment'], random_state=42, test_size=1/6, stratify=df2['sentiment']
    )
    val_text, test_text, val_labels, test_labels = train_test_split(
        temp_text, temp_labels, random_state=42, test_size=0.5, stratify=temp_labels
    )

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model)
    # Function to calculate length of text based on tokenizer's encoding
    def length_of_text(text):
        return len(tokenizer.encode(text, add_special_tokens=True))

    # Calculate the length of each text entry in train_text
    train_text_lengths = train_text.apply(length_of_text)

    # Calculate the average and maximum length of the texts in train_text
    average_length = train_text_lengths.mean()
    maximum_length = train_text_lengths.max()

    # Assuming you want to print it out or use it further in your code
    print(f'Average length of train_text: {average_length}')
    print(f'Maximum length of train_text: {maximum_length}')
    max_seq_len = 512

    # Tokenization and encoding of the datasets
    tokens_train = tokenizer.batch_encode_plus(
        train_text.tolist(), max_length=max_seq_len, pad_to_max_length=True,
        truncation=True, return_token_type_ids=False
    )
    tokens_val = tokenizer.batch_encode_plus(
        val_text.tolist(), max_length=max_seq_len, pad_to_max_length=True,
        truncation=True, return_token_type_ids=False
    )
    tokens_test = tokenizer.batch_encode_plus(
        test_text.tolist(), max_length=max_seq_len, pad_to_max_length=True,
        truncation=True, return_token_type_ids=False
    )

    # Convert to tensors
    train_seq = torch.tensor(tokens_train['input_ids'])
    train_mask = torch.tensor(tokens_train['attention_mask'])
    train_y = torch.tensor(train_labels.tolist())

    val_seq = torch.tensor(tokens_val['input_ids'])
    val_mask = torch.tensor(tokens_val['attention_mask'])
    val_y = torch.tensor(val_labels.tolist())

    test_seq = torch.tensor(tokens_test['input_ids'])
    test_mask = torch.tensor(tokens_test['attention_mask'])
    test_y = torch.tensor(test_labels.tolist())

    train_data = TensorDataset(train_seq, train_mask, train_y)
    val_data = TensorDataset(val_seq, val_mask, val_y)
    test_data = TensorDataset(test_seq, test_mask, test_y)
    
    return train_data, val_data, test_data, train_labels

def get_dataloader(train_data, val_data, test_data, batch_size=32):


    # DataLoaders
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # Return the DataLoaders and the labels
    return train_dataloader, val_dataloader, test_dataloader
