# Report Overview: Sentiment Classification with BERTweet

This project investigates sentiment classification using the pre-trained language model **BERTweet**, with a focus on reproduction and domain transfer. The study involves two phases: reproducing the original results on Twitter data and applying transfer learning to adapt the model to Reddit data.

---

## Architecture Overview

The architecture consists of three main components:

1. **Reproduction Phase**: Reimplementation of the original BERTweet-based sentiment classifier using the **SemEval-2017 Task 4** Twitter dataset.
2. **Domain Adaptation Phase**: Fine-tuning the reproduced model on Reddit sentiment data to evaluate generalization.
3. **Cross-Domain Evaluation**: Comparative performance analysis on a separate human-annotated Reddit dataset.

![Architecture](images/Architecture.png)

---

## Phase 1: Reproducing Twitter-Based Sentiment Classification

The first part of the project replicates the sentiment classification results reported in the following paper:

- **Paper**: [BERTweet: A Pre-trained Language Model for English Tweets (arXiv)](https://arxiv.org/abs/2005.10200)  
- **Dataset**: [SemEval-2017 Task 4](https://alt.qcri.org/semeval2017/task4/)

The reproduced model achieved an **F1-score of 71%**, which is reasonably close to the **78.2%** F1-score reported in the original publication.

---

## Phase 2: Transfer Learning from Twitter to Reddit

To evaluate the robustness of the BERTweet model across platforms, the classifier was fine-tuned on Reddit comments using the following dataset:

- **Dataset**: [Sentiment Analysis of Reddit Data (Kaggle)](https://www.kaggle.com/code/amarsharma768/sentiment-analysis-of-reddit-data)

The fine-tuned model achieved a **macro-average F1-score of 0.93**, indicating successful adaptation to a different linguistic domain.

---

## Phase 3: Evaluation on Human-Annotated Reddit Data

Both models were evaluated on a held-out Reddit dataset to assess cross-domain generalization:

- **Evaluation Dataset**: [Hugging Face: The Reddit Dataset (SocialGrep)](https://huggingface.co/datasets/SocialGrep/the-reddit-dataset-dataset)

The Reddit-fine-tuned model consistently outperformed the original Twitter-based model in terms of both **accuracy** and **F1-score**.

---

## Performance Summary

| Model              | Training Dataset                  | Evaluation Dataset         | F1-Score |
|--------------------|-----------------------------------|-----------------------------|----------|
| Reproduced         | SemEval-2017 Task 4 (Twitter)     | Reddit (SocialGrep)         | 0.39–0.40 |
| Reference (Paper)  | SemEval-2017 Task 4 (Twitter)     | Same                        | 0.782     |
| Transfer Learned   | Reddit (Kaggle)                   | Reddit (SocialGrep)         | 0.93      |

---

## References

1. Nguyen, D. Q., Vu, T., & Tuan Nguyen, A. (2020). [BERTweet: A Pre-trained Language Model for English Tweets](https://arxiv.org/abs/2005.10200). *arXiv preprint arXiv:2005.10200*.
2. [SemEval-2017 Task 4 Dataset](https://alt.qcri.org/semeval2017/task4/)  
3. [Reddit Training Dataset (Kaggle)](https://www.kaggle.com/code/amarsharma768/sentiment-analysis-of-reddit-data)  
4. [Reddit Testing Dataset (HuggingFace)](https://huggingface.co/datasets/SocialGrep/the-reddit-dataset-dataset)

---
