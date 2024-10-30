import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from train import train_model
from cross_validate import cross_validate_roberta
from evaluate import evaluate_model

model_path = "./best_roberta_news_classifier"
loaded_model = RobertaForSequenceClassification.from_pretrained(model_path)
loaded_tokenizer = RobertaTokenizer.from_pretrained(model_path)

def predict_category(headline, label_encoder, max_length=128):

    tokens = loaded_tokenizer(headline, padding='max_length', max_length=max_length, truncation=True, return_tensors='pt')
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokens = {key: val.to(device) for key, val in tokens.items()}

    loaded_model.to(device)
    loaded_model.eval()
    

    with torch.no_grad():
        outputs = loaded_model(**tokens)
        predicted_class = torch.argmax(outputs.logits, dim=-1)

    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())
    return predicted_label[0]


if __name__ == "__main__":

    import joblib
    label_encoder = joblib.load("label_encoder.joblib")  # Load your label encoder
    
    # Sample headline for prediction
    new_headline = ["New groundbreaking technology to revolutionize AI industry"]
    predicted_category = predict_category(new_headline, label_encoder)
    print(f"Predicted Category: {predicted_category}")
