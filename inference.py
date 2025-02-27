import os
import json
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Global variables
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def model_fn(model_dir):
    """Load the model for inference"""
    global tokenizer, model
    
    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    return model

def input_fn(request_body, request_content_type):
    """Parse input data"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """Make prediction with the loaded model"""
    # Tokenize input
    texts = input_data.get('texts', [])
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Move inputs to the same device as the model
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process outputs
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    predictions = torch.argmax(probabilities, dim=-1)
    
    # Return results
    return {
        'predictions': predictions.cpu().numpy().tolist(),
        'probabilities': probabilities.cpu().numpy().tolist()
    }

def output_fn(prediction, response_content_type):
    """Format the prediction response"""
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported content type: {response_content_type}")