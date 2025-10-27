# FIXED: utils/traditional_model.py - WITH MULTIPROCESSING + 2 classes only
# Replace your utils/traditional_model.py with this

import time
import numpy as np
from textblob import TextBlob
from multiprocessing import Pool, cpu_count
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob - BINARY CLASSIFICATION ONLY"""
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        
        # Binary classification: positive or negative only (no neutral)
        if polarity >= 0:
            return 'positive'
        else:
            return 'negative'
    except:
        return 'positive'

def process_chunk(text_chunk):
    """Process a chunk of texts (for multiprocessing)"""
    return [analyze_sentiment_textblob(text) for text in text_chunk]

def run_traditional_sentiment_analysis(texts, true_labels=None, num_processes=None):
    """
    Run sentiment analysis using TextBlob with multiprocessing
    Binary classification: positive/negative only
    """
    
    # Windows-safe multiprocessing
    if sys.platform.startswith('win'):
        num_processes = min(4, cpu_count())
    elif num_processes is None:
        num_processes = cpu_count()
    
    # For small datasets, use sequential
    if len(texts) < 100:
        start_time = time.time()
        predictions = [analyze_sentiment_textblob(text) for text in texts]
        processing_time = time.time() - start_time
    else:
        # Multiprocessing for larger datasets
        chunk_size = max(1, len(texts) // num_processes)
        text_chunks = [
            texts[i:i + chunk_size] 
            for i in range(0, len(texts), chunk_size)
        ]
        
        start_time = time.time()
        
        try:
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_chunk, text_chunks)
            predictions = [pred for chunk_result in results for pred in chunk_result]
        except Exception as e:
            print(f"Multiprocessing failed: {e}, using sequential...")
            predictions = [analyze_sentiment_textblob(text) for text in texts]
        
        processing_time = time.time() - start_time
    
    # Prepare results
    results_dict = {
        'predictions': predictions,
        'time': processing_time,
        'num_processes': num_processes if len(texts) >= 100 else 1
    }
    
    # Calculate accuracy if true labels provided
    if true_labels is not None:
        # Convert labels to binary (positive/negative only)
        true_labels_binary = []
        for label in true_labels[:len(predictions)]:
            label_str = str(label).lower().strip()
            # Map neutral to positive
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
            results_dict['accuracy'] = 0.0
    
    return results_dict
