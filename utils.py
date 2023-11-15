import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import os
from datetime import datetime

def count_label_ratio(df):
    # Count the occurrences of each unique value in the 'sentiment' column
    value_counts = df['sentiment'].value_counts()

    # Get the counts for 'positive' and 'negative' sentiments
    count_positive = value_counts.get('positive', 0)  # Returns 0 if 'positive' is not in the index
    count_negative = value_counts.get('negative', 0)  # Returns 0 if 'negative' is not in the index

    # Calculate the ratio of 'positive' to 'negative'
    if count_negative == 0:  # To avoid division by zero
        ratio = float('inf')  # Set the ratio to infinity
    else:
        ratio = count_positive / count_negative

    print(f"Count of positive sentiments: {count_positive}")
    print(f"Count of negative sentiments: {count_negative}")
    print(f"Ratio of positive to negative sentiments: {ratio}")


def compute_acc(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  # No gradients needed
        for i, batch in enumerate(dataloader):
            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            sent_id, mask, labels = batch
            
            labels.to(device)

            # Forward pass to get outputs
            outputs = model(sent_id, mask)

            # Convert outputs probabilities to predicted class (0 or 1)
            preds = torch.argmax(outputs, dim=1)

            # Compare with true labels and update correct predictions
            correct_predictions += torch.sum(preds == labels).item()
            total_predictions += labels.size(0)

    # Calculate the accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy

# Set the parameters
def set_params(lr=-1, bs=-1, dropout=-1, epochs=-1, frozen = -1):
    params = {}
    
    # Set or randomly select the learning rate
    params['lr'] = lr if lr != -1 else random.choice([5e-5, 3e-5, 2e-5])
    
    # Set or randomly select the batch size
    params['bs'] = bs if bs != -1 else random.choice([16, 32])
    
    # Set or randomly select the dropout rate
    params['dropout'] = dropout if dropout != -1 else random.choice([0, 0.1])
    
    # Set or randomly select the number of epochs
    params['epochs'] = epochs if epochs != -1 else random.choice([2, 3, 4])

    # Set the frozen parameter
    if frozen == -1:
        params['frozen'] = {
        'encoder': random.choice([0, 4, 8, 12]),
        'embeddings': random.choice([False, True]),
        'pooler': random.choice([False, True])
    }
    else:
        params['frozen'] = frozen
    
    return params

def plot_losses(train_losses, valid_losses, params):
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Creating a unique filename based on timestamp
    filename = f"results/loss_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    # Plotting the losses
    plt.figure(figsize=(10, 6))
    # Plotting the losses with epochs starting from 1
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation Loss', marker='x')

    plt.title(f"Training and Validation Losses\n"
              f"lr={params['lr']}, bs={params['bs']}, "
              f"dropout={params['dropout']}, epochs={params['epochs']}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Saving the plot
    plt.savefig(filename)
    

