# Preference Prediction between Large Language Models

This project aims to predict **human preferences between responses from two Large Language Models (LLMs)**. By modeling user preferences, we help bridge the gap between LLM capabilities and actual human expectations, contributing to better alignment in chatbot systems.

---

## 📌 Motivation

This task is closely related to **Reinforcement Learning from Human Feedback (RLHF)**. Accurately predicting which model response a user prefers enables:

- Training better aligned models
- Reducing hallucinations and irrelevant outputs
- Designing human-centric assistant systems

---

## 📂 Dataset Overview

- Source: **Human preference judgments**
- Training size: **55,000** samples  
- Test size: **25,000** samples  
- Total models involved: **64 LLMs**

### 📊 Columns

| Column             | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `id`               | Unique identifier                                            |
| `prompt`           | Input prompt given to both models                            |
| `response_a/b`     | Responses from model A and model B respectively              |
| `winner_model_[a/b/tie]` | Binary indicator of judge's preferred response        |

> 🧩 31% of all comparisons are ties  
> 🏆 Most frequent matchups:
> - `gpt-4-1106-preview` vs `claude-2.1`
> - `gpt-4-1106-preview` vs `gpt-4-0613`
> - `claude-1` vs `claude-2.1`

---

## 🧠 Task

**Input**:  
- A prompt + two model responses

**Output**:  
- Predicted probabilities:
  - Model A is preferred
  - Model B is preferred
  - Tie

**Evaluation Metric**:  
- **Multiclass Log Loss**  
<img width="504" height="72" alt="image" src="https://github.com/user-attachments/assets/fb5cca18-0fe8-433f-812f-72e30a3dc069" />


---

## 🔍 Data Preprocessing

- **Text Cleaning**:  
  - Lowercasing, removing digits and punctuation
  - Tokenization using `nltk`
  - Stopword removal & Lemmatization

- **Input Formatting**:  
  - Concatenate prompt and each response into input pairs
  - Label encoding:
    - Model A preferred → `0`
    - Model B preferred → `1`
    - Tie → `2`

---

## 🏗️ Model Architecture

- Backbone: `DeBERTa-v3-small` or `distilbert-base-uncased`
- Input: `(prompt, response_a)` and `(prompt, response_b)`
- Processing:
  - Tokenized separately and passed through shared encoder
  - Mean pooling over final hidden states
  - Concatenation of both vectors
  - Fully connected layer + Softmax over 3 classes

---

## ⚙️ Training Setup

- **Loss**: Custom `DPO Loss` (Direct Preference Optimization)  
- **Optimizer**: AdamW  
- **Learning rate**: `2e-5`  
- **Max length**: `512`  
- **Epochs**: `3`  
- **Batch size**: `16`  
- **Mixed precision**: Enabled via `torch.cuda.amp`

---

## 📈 Results

| Logits Pre-Softmax | Inference Softmax | Epoch | Score (LogLoss) |
|--------------------|-------------------|--------|------------------|
| ❌                  | ✅                | 1      | 1.04219          |
| ❌                  | ✅                | 3      | 1.07467          |
| ✅ (default)        | ✅                | 1      | 1.04398          |
| ✅ (log_softmax)    | ✅                | 1      | 1.07435          |
| ✅                  | ❌                | 1      | 1.35201          |
| ✅                  | ❌                | 3      | 1.58984          |

> ✅ Best result: **LogLoss = 1.04398**, with softmax applied only during inference

---

## 🧪 Inference

- Outputs probability distribution over 3 classes (A win / B win / Tie)
- Saves result to `submission.csv` with columns:
  - `winner_model_a`
  - `winner_model_b`
  - `winner_tie`

---

## 🔮 Future Work

- Improve generalization through:
  - More diverse augmentation
  - Multi-turn dialogue structure
- Combine outputs from **multiple LLMs** to reduce bias
- Try **contrastive learning** or **pairwise ranking losses**
- Apply **LoRA** or **PEFT** fine-tuning techniques
- Evaluate **explainability** with attention or SHAP
