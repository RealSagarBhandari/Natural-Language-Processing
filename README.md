# Natural Language Processing 

[![GitHub](https://img.shields.io/badge/GitHub-RealSagarBhandari-181717?logo=github)](https://github.com/RealSagarBhandari/Natural-Language-Processing)
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)


This directory contains a collection of Jupyter Notebooks exploring essential concepts in Natural Language Processing (NLP), Large Language Models (LLMs), tokenization strategies, various Models , and many more . 


Also check out :  [Machine Learning](https://github.com/RealSagarBhandari/Machine-Learning)



**Below is the documentation:**




##  1.  [A1_HuggingFace](https://github.com/RealSagarBhandari/Natural-Language-Processing/blob/main/A1_HuggingFace.ipynb)
##  Transformers: Deep Dive into Fine-Grained Sentiment & Emotion Analysis


[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-yellow)](https://huggingface.co/docs/transformers/index)





## 📌 Project Overview
This project is a technical exploration of the **Hugging Face Transformers** ecosystem. It transitions from high-level pipeline implementation to a comparative analysis of specialized architectures, including **BERT**, **RoBERTa**, and **DistilRoBERTa**.

The core objective was to move beyond binary sentiment classification and explore high-dimensional emotion detection (28 distinct categories) and domain-specific fine-tuning for finance, social media, and security.


* **Model Families:** BERT (Google), RoBERTa (Meta), and DistilRoBERTa



### 1. Emotion Intelligence: `roberta-base-go_emotions`
* **Architecture:** RoBERTa (Robustly Optimized BERT Pretraining Approach), an improved version of BERT.
* **Granularity:** Identifies 28 labels (27 specific emotions + 1 neutral).
* **Data Source:** Trained on 58,000 manually labeled Reddit comments.
* **Training Depth:** Utilizes a 160GB corpus including BookCorpus, English Wikipedia, and CC-News.


### 2. Specialized Classifiers
| Model | Task | Source/Benchmark |
| :--- | :--- | :--- |
| **Twitter-RoBERTa** | Social Media Sentiment | 124M Tweets (2018-2021) |
| **FinBERT** | Financial Sentiment | Financial PhraseBank (98.23% accuracy) |
| **RoBERTa-Spam** | Security Detection | Enron Email, Telegram, and SMS datasets |
| **BERT-AG-News** | Topic Classification | AG News (World, Sports, Tech, Business) |

---

##  Key Technical Takeaways
* **Transfer Learning Efficiency:** Demonstrated how pre-trained "base" models are fine-tuned into specialized experts for niche domains like finance or security.
* **Hardware Acceleration:** Leveraged the `accelerate` library to automatically detect and utilize available hardware (MPS/GPU), significantly reducing inference latency.
* **Evolution of Architectures:** Analyzed the "family tree" from the original BERT (2018) to the more robustly optimized RoBERTa (2019) and efficient DistilRoBERTa.



## 2. [A2_DataEvaluation](https://github.com/RealSagarBhandari/Natural-Language-Processing/blob/main/A2_DataEvaluation.ipynb)




## NLP Model Evaluation & Text Classification


## 📌 Overview
This repository contains a comprehensive data science and machine learning evaluation pipeline focused on **Natural Language Processing (NLP)**. The project demonstrates how to load, explore, and process large-scale datasets from the Hugging Face Hub, and critically evaluate the performance of pre-trained Transformer models on complex text classification tasks. 

Instead of relying solely on baseline accuracy, this project dives deep into **advanced classification metrics** to uncover the true predictive power and failure modes of modern language models.




---

##  Key Features
- **Scalable Data Integration:** Efficiently loads and streams large datasets (`DatasetDict`, `Apache Arrow` lazy-loading/zero-copy formats) directly from the Hugging Face Hub.
- **Pre-trained Transformers:** Utilizes state-of-the-art RoBERTa models (`cardiffnlp/twitter-roberta-base-sentiment-latest` and `SamLowe/roberta-base-go_emotions`) via the `transformers` pipeline for inference.
- **Rigorous Evaluation:** Bypasses basic accuracy metrics to implement comprehensive evaluation suites using `scikit-learn`:
  - Macro & Weighted **Precision, Recall, and F1-Scores**.
  - **Classification Reports** for highly granular multi-class outputs.
  - **Confusion Matrices** to visually diagnose model misclassifications and label overlaps.

---

## Datasets Analyzed
1. **[GoEmotions](https://huggingface.co/datasets/go_emotions):** A massive dataset of Reddit comments labeled across 28 categories (27 distinct emotions + neutral). Represents a highly complex, multi-label text classification problem.
2. **[BBC News (`SetFit/bbc-news`)](https://huggingface.co/datasets/SetFit/bbc-news):** A multi-class dataset categorizing news articles into distinct topics (business, entertainment, politics, sport, tech).

---

## Models Utilized
- **Twitter RoBERTa Base Sentiment:** Explored for general positive/negative/neutral sentiment extraction.
- **GoEmotions RoBERTa:** A specialized model mapped directly to the 28 emotion categories to test fine-tuned predictive capability against ground-truth labels.

---

## Key Insights & Takeaways
1. **Evaluation Metrics Tell a Story:** Accuracy alone is inherently flawed, especially in imbalanced datasets (e.g., simply guessing the most common label, "neutral", yields an artificial ~30% baseline accuracy). Precision, recall, and F1-scores are critical for assessing true model reliability.
2. **Diagnosing Model Blindspots:** The confusion matrix reveals exactly where the model struggles (e.g., misclassifying nuanced emotions like *annoyance* vs *anger*, or *realization*). Balanced metrics (precision ≈ recall) indicate robust training.
3. **Task Complexity Matters:** Differentiating between 2 sentiment labels (e.g., positive vs. negative movie reviews) versus 28 granular emotion labels requires vastly different analytical approaches and model specializations.

---




## 3. [A3_Rouge_Matrics_and_Summarization.ipynb](https://github.com/RealSagarBhandari/Natural-Language-Processing/blob/main/A3_Rouge_Matrics_and_Summarization.ipynb)


## NLP Text Summarization & ROUGE Evaluation



##  Overview
This repository features a rigorous empirical evaluation of Natural Language Processing (NLP) text summarization models. Moving beyond basic implementation, this project critically analyzes the **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** metric across diverse, large-scale datasets. 

The pipeline demonstrates how dataset design, summary style (extractive vs. abstractive), and evaluation metrics fundamentally interact, providing deep insights into the true performance and limitations of modern language models on compression tasks.



---

##  Key Capabilities & Technical Scope
* **Automated Evaluation Pipelines:** Implements programmatic ROUGE scoring (ROUGE-1, ROUGE-2, ROUGE-L) to systematically benchmark model outputs against human-written reference summaries.
* **Cross-Domain Analysis:** Tests summarization performance across drastically different textual domains (news media vs. dense legislative documents).
* **Critical Metric Assessment:** Exposes the inherent limitations of n-gram overlap metrics when evaluating highly abstractive, conceptually compressed text.

---

##  Datasets Evaluated
To test the models across a spectrum of difficulty and linguistic styles, the following Hugging Face datasets were utilized:
1.  **[CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail):** Standard news articles. Tends to favor *extractive* summarization with higher lexical overlap.
2.  **[BillSum](https://huggingface.co/datasets/billsum):** US Congressional and California state bills. Tests the model's ability to parse and summarize dense, highly technical legislative language.
3.  **[XSum (Extreme Summarization)](https://huggingface.co/datasets/xsum):** BBC articles paired with highly compressed, single-sentence summaries. Requires extreme *abstractive* capabilities.

---

##  Empirical Insights & Findings
The evaluation revealed a clear hierarchy in ROUGE performance based on dataset architecture:

**ROUGE Score Ranking: `CNN/DailyMail > Billsum > XSum`**

* **The Lexical Overlap Bias:** Datasets that naturally allow for more extractive summaries (like CNN/DailyMail) produced significantly higher ROUGE scores. 
* **The Abstraction Penalty:** XSum yielded the lowest scores, not necessarily due to poor model performance, but because the dataset requires extreme abstractive compression. ROUGE penalizes valid conceptual summaries if they lack direct n-gram overlap with the reference text.
* **Takeaway:** ROUGE is highly sensitive to dataset design. High scores do not universally equate to "better" summarization; they often just indicate higher extractive overlap. Evaluation must always be contextualized by the specific abstraction requirements of the task.

---



## 4.  [B1_Chat_Conversation_and_Models.ipynb](https://github.com/RealSagarBhandari/Natural-Language-Processing/blob/main/B1_Chat_Conversation_and_Models.ipynb)



## SLM Conversational AI & Model Benchmarking

##  Overview
This repository focuses on the implementation and rigorous evaluation of **Small Language Models (SLMs)** for conversational AI. While massive LLMs dominate the landscape, SLMs offer a highly efficient, edge-deployable alternative. 

This project demonstrates how to build a multi-turn chatbot using the Hugging Face `text-generation` pipeline, manage conversational memory (state), and construct custom empirical benchmarks to test reasoning, factual retrieval, and instruction-following limits in sub-billion parameter models.


---

##  Key Capabilities & Technical Scope

* **Multi-Turn State Management:** Engineered a continuous conversational loop that dynamically updates and passes chat history (system, user, and assistant roles) to maintain context across multiple interactions.
* **Pipeline Implementation:** Leveraged Hugging Face's `pipeline` and `accelerate` libraries for optimized model loading and hardware-accelerated inference.
* **Custom Benchmarking Suite:** Designed an adversarial prompt evaluation framework ("Animal Trivia & Jokes") designed to test specific model vulnerabilities:
  * *Formatting Constraints* (e.g., "exactly one sentence").
  * *Logical Reasoning vs. Hallucination* (e.g., assessing impossible scenarios like a bear vs. shark playing chess).
  * *Factual Retrieval Accuracy.*

---

##  Models Evaluated
The project directly compares two highly optimized instruct-tuned SLMs:
1. **[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct):** A highly capable 0.5 billion parameter model.
2. **[HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct):** A lightweight 360 million parameter model designed for extreme efficiency.

---

##  Empirical Insights & Findings
Testing revealed fascinating behaviors and limitations inherent to sub-billion parameter architectures:

* **Instruction Adherence:** Smaller models struggle with strict formatting constraints. While factual retrieval (e.g., cheetah top speeds) is generally accurate, instructions like "reply in exactly one sentence" frequently result in run-on sentences or multi-sentence failures.
* **Reasoning vs. Hallucination:** When faced with abstract or nonsensical reasoning tasks (the bear vs. shark chess match), both models struggled to break out of their training distribution. Instead of logically deducing the physical impossibility of the scenario, both models hallucinated complex, anthropomorphized strategies or injected unrelated human elements.
* **Overall Verdict:** While SLMs are remarkably capable for basic text generation and localized factual retrieval, they require heavy guardrails for logic-based or strictly formatted tasks. In this specific benchmark, **Qwen2.5-0.5B-Instruct** demonstrated a slight edge in strict instruction following over SmolLM2.

---


