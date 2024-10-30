import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import RobertaForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
from train import train_model


def cross_validate_roberta(df, label_encoder, batch_size=16, num_epochs=3, k_folds=5, learning_rate=2e-5):
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])

    y_train_val_encoded = label_encoder.fit_transform(train_val_df['category'])
    y_test_encoded = label_encoder.transform(test_df['category'])

    X_train_val_tokens = tokenize_data(train_val_df['clean_headline'])
    X_test_tokens = tokenize_data(test_df['clean_headline'])

    train_val_dataset = TensorDataset(X_train_val_tokens['input_ids'], X_train_val_tokens['attention_mask'], torch.tensor(y_train_val_encoded))
    test_dataset = TensorDataset(X_test_tokens['input_ids'], X_test_tokens['attention_mask'], torch.tensor(y_test_encoded))

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    accuracy_list = []
    f1_list = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
        print(f"\nTraining Fold {fold + 1}/{k_folds}")

        
        train_subset = torch.utils.data.Subset(train_val_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_val_dataset, val_idx)
        
        train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch_size)

        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(label_encoder.classes_))
        model.to(device)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        
        model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            progress_bar = tqdm(train_dataloader, desc="Training")

            for batch in progress_bar:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                optimizer.zero_grad()

                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Backward pass
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({"Loss": loss.item()})
    
        model.eval()
        val_labels = []
        val_predictions = []

        for batch in val_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)

                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_predictions)
        f1 = f1_score(val_labels, val_predictions, average='weighted')

        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
        
        accuracy_list.append(accuracy)
        f1_list.append(f1)

    tokenizer.save_pretrained("./best_roberta_news_classifier")

    avg_accuracy = np.mean(accuracy_list)
    avg_f1 = np.mean(f1_list)

    print(f"\nCross-Validation Results - Average Accuracy: {avg_accuracy:.4f}, Average F1 Score: {avg_f1:.4f}")

    return avg_accuracy, avg_f1
