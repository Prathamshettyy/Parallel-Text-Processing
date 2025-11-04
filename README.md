# Parallel Text Processing - Sentiment Analysis

## Project Overview

Parallel Text Processing is a multi-core sentiment analysis system that implements concurrent text processing using Python's multiprocessing architecture. The platform provides dual-method sentiment classification combining rule-based (TextBlob) and deep learning (DistilBERT) approaches, enabling organizations to process large-scale text datasets efficiently while comparing speed and accuracy trade-offs.

## Project Details

### Architecture

The system consists of five integrated modules:

1. **Data Processing Module** - Parallel text cleaning, tokenization, lemmatization, and keyword extraction
2. **Traditional Model** - TextBlob-based sentiment analysis with multiprocessing
3. **LLM Model** - DistilBERT transformer model with GPU acceleration
4. **Email Service** - SMTP-based result distribution with CSV attachments
5. **Web Interface** - Streamlit application with user authentication and real-time visualization

### Technology Stack

- Python 3.8+, Streamlit, TextBlob, Transformers, PyTorch, NLTK, SQLite3, Scikit-learn, Pandas

### Key Features

- Parallel processing across all available CPU cores
- Flexible column detection supporting diverse datasets
- Real-time performance comparison metrics
- Confusion matrix visualization
- Email result distribution
- User authentication with SQLite backend

## Experimental Results

### Test Configuration

Dataset: 1000 sentiment-labeled text samples
Processing: Parallel execution across available CPU cores
Evaluation: Supervised classification with labeled data

### Performance Metrics

#### Speed Analysis

| Metric | TextBlob | DistilBERT | Difference |
|--------|---------|-----------|-----------|
| Processing Time | 21.57s | 42.64s | +21.07s (+97.6%) |
| Throughput | 46.4 texts/sec | 23.4 texts/sec | 50% reduction |
| Speed Advantage | 1.97x faster | Baseline | DistilBERT 1.97x slower |

TextBlob processes data almost twice as fast due to its rule-based approach that requires no neural network inference or model loading overhead.

#### Accuracy Analysis

| Metric | TextBlob | DistilBERT | Improvement |
|--------|---------|-----------|------------|
| Overall Accuracy | 69.4% | 80.5% | +11.1% |
| Accuracy Gain | Baseline | +11.1 points | 16% improvement |
| Performance | Standard | Superior | Better classification |

DistilBERT delivers 11.1 percentage points higher accuracy through contextual understanding of text semantics.

#### Confusion Matrix Results

**TextBlob (Traditional Method):**
- True Negatives: 250 | False Positives: 274
- False Negatives: 0 | True Positives: 444
- Classification Pattern: Strong positive bias with 100% recall but 61.8% precision
- Characteristics: Catches all positive sentiments but misclassifies 9.7% of negatives

**DistilBERT (LLM Method):**
- Improved negative detection and reduced false positives
- Better precision-recall balance
- More nuanced classification of complex sentiments
- Superior handling of context-dependent expressions

### Comparative Analysis

#### Why TextBlob is Faster (21.57 seconds vs 42.64 seconds)

1. Rule-Based Approach: Pre-defined polarity scoring without neural computation
2. Minimal Overhead: Direct mathematical operations on text features
3. No Model Loading: Eliminates transformer initialization (5-10 seconds typical)
4. CPU-Only: Optimized single-threaded execution
5. Lightweight Algorithm: Simple sentiment calculation without backpropagation

**Result:** 1.97x faster processing, suitable for real-time applications requiring high throughput

#### Why DistilBERT Has Better Accuracy (80.5% vs 69.4%)

1. Contextual Understanding: Transformer architecture captures word relationships
2. Pre-trained Knowledge: Billions of parameters trained on massive text corpora
3. Attention Mechanisms: Dynamic weighting of important words based on context
4. Nuance Detection: Understands sarcasm, negation, and complex expressions
5. Deep Representations: Learned features exceed rule-based heuristics

**Result:** 11.1% higher accuracy, optimal for precision-critical applications

### Speed-Accuracy Trade-off

| Factor | TextBlob | DistilBERT | Recommendation |
|--------|---------|-----------|--------------|
| Latency | Low | High | TextBlob for real-time |
| Accuracy | Moderate | High | DistilBERT for precision |
| Resource | Minimal | Significant | TextBlob for constraints |
| Throughput | High | Moderate | TextBlob for volume |
| Complexity | Simple | Advanced | DistilBERT for nuance |

## Use Case Selection Guide

### Use TextBlob When:
- Real-time processing required (< 30 seconds for 1000 texts)
- High throughput needed (46+ texts/second)
- Resource-constrained environments
- Speed prioritized over accuracy
- Applications: Live monitoring, stream processing, high-volume batch jobs

### Use DistilBERT When:
- Accuracy critical (80%+ required)
- Complex sentiment understanding needed
- Processing time not a constraint
- Adequate computational resources available
- Applications: Product reviews, customer feedback, market analysis

### Hybrid Approach:
- Initial TextBlob filtering for fast classification
- Deep DistilBERT analysis of uncertain cases
- Combined results for balanced performance

## Installation and Execution

```bash
git clone https://github.com/Prathamshettyy/Parallel-Text-Processing.git
cd Parallel-Text-Processing
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Key Findings

The comparative analysis validates fundamental principles in machine learning:

1. Rule-based systems prioritize speed through simplicity
2. Deep learning models achieve accuracy through learned complexity
3. Parallel processing enables scalable solutions for both approaches
4. Contextual understanding significantly improves sentiment classification
5. Strategic selection based on use case requirements optimizes outcomes

## Conclusion

Parallel Text Processing demonstrates that efficient sentiment analysis requires strategic algorithm selection. TextBlob excels in high-speed scenarios with 21.57 seconds for 1000 texts, while DistilBERT delivers superior accuracy at 80.5%. Organizations can optimize performance by selecting appropriate methods based on latency requirements, accuracy thresholds, and resource availability. The system's parallel architecture enables scalable deployment for both traditional and deep learning approaches across enterprise environments.
