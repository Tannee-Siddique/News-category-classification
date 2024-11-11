import torch
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score
from cross_validate import cross_validate_roberta


def evaluate_model(test_dataset, model_path, batch_size=16):

    model = RobertaForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    model.eval()

   
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    test_labels = []
    test_predictions = []

    
    for batch in test_dataloader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        with torch.no_grad():  
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(predictions.cpu().numpy())

    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_f1 = f1_score(test_labels, test_predictions, average='weighted')

    print(f"Test Set Accuracy: {test_accuracy:.4f}")
    print(f"Test Set F1 Score: {test_f1:.4f}")

    return test_accuracy, test_f1
