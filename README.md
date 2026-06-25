# Arabic Text Summarization Using Seq2Seq with Attention

## Overview
This project focuses on Arabic abstractive text summarization using a custom Seq2Seq deep learning model with Bahdanau Attention. The goal is to generate short Arabic headlines or summaries from full Arabic news articles.

Arabic summarization is challenging because Arabic has rich morphology, right-to-left writing, different dialects, and many word forms.

## Project Objectives
- Build an Arabic text summarization system.
- Merge and prepare multiple Arabic summarization datasets.
- Clean and preprocess Arabic text.
- Tokenize Arabic text using the AraBART tokenizer.
- Build a custom Seq2Seq model using PyTorch.
- Use a Bidirectional LSTM encoder.
- Use an LSTM decoder with Bahdanau Attention.
- Train the model with early stopping.
- Evaluate the model using ROUGE-1 and ROUGE-L.

## Datasets
The project uses four Arabic news summarization datasets:

| Dataset | Format | Notes |
|---|---|---|
| SumArabic 1.0 | JSONL | Arabic news with text and headline |
| AraSum | TSV | Large Arabic news summarization corpus |
| Egyptian News | Parquet | Egyptian dialect news summaries |
| Arabic Kaggle | CSV | Arabic summarization dataset |

The final merged training set contained **46,911 article-headline pairs**, with validation and test sets containing around **8,000+ pairs each**.

## Data Preprocessing
The preprocessing pipeline included:
- Removing extra whitespace
- Removing unwanted quotation marks
- Removing Arabic guillemets
- Removing Latin characters
- Removing noisy parenthetical numbers
- Removing isolated Arabic letters
- Removing newlines
- Removing Unicode control characters
- Normalizing punctuation

## Tokenization
The project used the **moussaKam/AraBART** tokenizer from HuggingFace Transformers.

Tokenization settings:
- Source max length: 1024 tokens
- Target max length: 256 tokens
- Training encoder length: 256 tokens
- Training decoder length: 64 tokens

## Model Architecture
The model is a custom Seq2Seq architecture built from scratch using PyTorch.

Main components:
- Encoder embedding layer
- Bidirectional LSTM encoder
- Decoder embedding layer
- LSTM decoder
- Bahdanau attention mechanism
- Fully connected output layer

## Hyperparameters
| Hyperparameter | Value |
|---|---|
| Embedding dimension | 128 |
| Hidden size | 256 |
| Batch size | 32 |
| Max epochs | 10 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Early stopping patience | 3 epochs |

## Evaluation
The model was evaluated using:
- ROUGE-1
- ROUGE-L

Evaluation was performed on 50 randomly selected validation samples.

## Files in This Repository
```text
Arabic-Text-Summarization-Seq2Seq-Attention/
│
├── NLP_Report_.pdf
├── PreProcessing_NLP (2).ipynb
├── final_full_seq2seq_early_stopping (1).ipynb
└── README.md
