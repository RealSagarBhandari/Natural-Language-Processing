
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

