
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaForSequenceClassification, AdamW
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
from transformers import get_scheduler
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd

# Load the preprocessed data
data = pd.read_csv("../results/preprocessed_data.csv")

# Access necessary columns for training
X = data[['cleaned_headline', 'cleaned_short_description', 'headline_tokens', 'description_tokens']]
y = data['category_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_data(text_series, max_length=128):
    # Ensure input is a list of strings
    return tokenizer(
        text_series.tolist(),  # Convert Series to a list of strings
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )


# Tokenize the training and testing data
X_train_tokens = tokenize_data(X_train['cleaned_headline'])
X_test_tokens = tokenize_data(X_test['cleaned_headline'])




def train_model(X_train_tokens, X_test_tokens, y_train, y_test, batch_size=16, num_epochs=3, learning_rate=2e-5):
    # Encode the labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Convert data to TensorDataset
    train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], torch.tensor(y_train_encoded))
    test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], torch.tensor(y_test_encoded))

    # Define data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Load the pre-trained RoBERTa model
    num_labels = len(label_encoder.classes_)  
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=num_labels)

    # Move the model to GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training parameters
    total_steps = len(train_dataloader) * num_epochs

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Training loop
    model.train()

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Unpack the batch and move to the device
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            lr_scheduler.step()

            progress_bar.set_postfix({"Loss": loss.item()})

    # Save the model after training
    model.save_pretrained("./roberta-news-category-classifier")
    print("Model saved to ./roberta-news-category-classifier")








