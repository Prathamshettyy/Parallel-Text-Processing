import time
import torch
import numpy as np
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def run_llm_sentiment_analysis(texts, true_labels=None, batch_size=32, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Run sentiment analysis using DistilBERT with GPU
    Binary classification: positive/negative only
    """
    
    # Check for GPU
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    
    print(f"\n{'='*50}")
    print(f"Device: {device_name}")
    if device == 0:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU - Install CUDA PyTorch for speed")
    print(f"{'='*50}\n")
    
    # Adjust batch size for CPU
    if device == -1:
        batch_size = 8
    
    # Load model
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
            truncation=True,
            max_length=512
        )
        print(f"✓ Model loaded on {device_name}")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Falling back to CPU...")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1,
            truncation=True,
            max_length=512
        )
        device_name = "CPU"
        batch_size = 8
    
    # Process
    start_time = time.time()
    
    try:
        print(f"Processing {len(texts)} texts (batch_size={batch_size})...")
        results = sentiment_pipeline(
            texts,
            batch_size=batch_size,
            truncation=True,
            max_length=512
        )
    except Exception as e:
        print(f"Error: {e}")
        print("Reducing batch size...")
        results = sentiment_pipeline(
            texts,
            batch_size=4,
            truncation=True,
            max_length=512
        )
    
    processing_time = time.time() - start_time
    print(f"✓ Completed in {processing_time:.2f}s\n")
    
    # Extract predictions - BINARY ONLY (positive/negative)
    predictions = []
    confidence_scores = []
    
    for result in results:
        label = result['label'].upper()
        score = result['score']
        
        # DistilBERT outputs: POSITIVE or NEGATIVE
        if label == 'POSITIVE':
            predictions.append('positive')
        else:  # NEGATIVE
            predictions.append('negative')
        
        confidence_scores.append(score)
    
    # Results
    results_dict = {
        'predictions': predictions,
        'confidence_scores': confidence_scores,
        'time': processing_time,
        'device': device_name,
        'batch_size': batch_size,
        'model': model_name
    }
    
    # Accuracy if labels provided
    if true_labels is not None:
        # Convert to binary
        true_labels_binary = []
        for label in true_labels[:len(predictions)]:
            label_str = str(label).lower().strip()
            if label_str in ['positive', 'pos', '1', 'good']:
                true_labels_binary.append('positive')
            elif label_str in ['negative', 'neg', '0', 'bad']:
                true_labels_binary.append('negative')
            else:  # neutral or unknown -> positive
                true_labels_binary.append('positive')
        
        try:
            accuracy = accuracy_score(true_labels_binary, predictions)
            results_dict['accuracy'] = accuracy
            
            # Binary classification report
            labels = ['negative', 'positive']
            report = classification_report(true_labels_binary, predictions,
                                          labels=labels, zero_division=0)
            results_dict['classification_report'] = report
            
            # Binary confusion matrix (2x2 = 4 boxes)
            cm = confusion_matrix(true_labels_binary, predictions, labels=labels)
            results_dict['confusion_matrix'] = cm
            results_dict['confusion_matrix_labels'] = labels
            
        except Exception as e:
            print(f"Could not calculate metrics: {e}")
    
    return results_dict
