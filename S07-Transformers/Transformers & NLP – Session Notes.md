# Transformers & NLP – General Notes

## **1. What to Remember**

* **Transformers solve sequence modeling limitations** of RNNs/LSTMs:
  * Can handle **long-range dependencies** without vanishing gradients.
  * Fully **parallelizable**, unlike RNNs.
  * Scales well for large datasets (BERT, GPT, etc.).
* **Key Transformer components**:
  * **Embedding Layer**: Converts tokens to dense vectors.
  * **Positional Encoding (PE)**: Adds order information to token embeddings.
  * **Self-Attention / Multi-Head Attention**:
    * Calculates relationships between tokens in a sequence.
    * Formula: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V`
    * **Multi-head** allows capturing **different types of relationships** simultaneously.
  * **Feed-Forward Network (FFN)**:
    * Applies non-linear transformation to each token individually.
    * Typically: `FFN(x) = max(0, xW1 + b1) W2 + b2` (2-layer MLP per token).
  * **Layer Normalization & Residual Connections**:
    * Stabilizes training and allows deep stacking.
* **Transformer Variants**:
  * **Encoder-only**: BERT – for understanding tasks (classification, QA).
  * **Decoder-only**: GPT – for generation tasks (text completion).
  * **Encoder-Decoder**: T5, BART – for seq-to-seq tasks (translation, summarization).
* **Attention visualization tip**: Always show **query, key, value** flow in your diagrams; students often get confused otherwise.

---

## **2. Complete NLP Pipeline**

1. **Data Collection**
   * Gather raw text (tweets, articles, books, etc.).
   * Ensure **clean and diverse dataset** for better model learning.
2. **Preprocessing**
   * **Cleaning**: Lowercasing, punctuation removal (optional depending on model), removing unwanted characters.
   * **Special tokens**: `[CLS]`, `[SEP]`, `[PAD]` depending on the model.
3. **Tokenization**
   * **Convert raw text into tokens** that the model understands.
     * Word-level, subword-level (BPE, WordPiece, SentencePiece), or character-level.
   * **Map tokens to IDs** using a vocabulary.
   * **Add attention masks** to indicate padding positions.
   * Optional: **Truncation and padding** to fixed sequence length.
4. **Embedding Layer**
   * Converts token IDs into **dense vectors** (`d_model` dimensions).
   * Includes **positional encoding** to inject sequence order information.
5. **Transformer Model**
   * **Input → Encoder/Decoder stacks → Output representations**
   * Internal flow:
     1. Embeddings + Positional Encoding → Multi-Head Self-Attention
     2. Attention output → Add & Norm → Feed-Forward → Add & Norm
     3. Stack `N` layers
   * **Output**:
     * Encoder: contextual embeddings for each token
     * Decoder: generates sequences autoregressively (for GPT / translation)
6. **Task-Specific Head**
   * Classification: softmax over classes
   * Generation: linear layer + softmax over vocabulary
   * Token tagging (NER/POS): softmax per token
7. **Training / Fine-Tuning**
   * Pretrained models are **adapted to your specific task**.
   * Uses task-specific loss:
     * Cross-entropy for classification
     * Language modeling loss for generation / translation
8. **Evaluation**
   * Accuracy, F1, BLEU, ROUGE depending on the task.
   * Check for **overfitting** and **generalization**.
9. **Inference / Deployment**
   * Use model to **predict new unseen data**.
   * Optional: **Quantization/Optimization** for faster inference.

> The sentence 'I love machine learning' is tokenized into ['I', 'love', 'machine', 'learning'], converted into token IDs [38, 2345, 9876, 12345], and then each ID is transformed into a dense embedding vector to be processed by the Transformer.

---

## **3. Fine-Tuning Explained**

Fine-tuning is the process of **adapting a pretrained Transformer** to a specific downstream task.

### **Why Fine-Tuning?**

* Pretrained models (GPT, BERT) already **understand language patterns**.
* Fine-tuning allows the model to **learn your specific task** without training from scratch.

### **Steps to Fine-Tune**

1. **Select a Pretrained Model**
   * Example: `bert-base-uncased` for classification, `gpt-3` for text generation.
2. **Prepare Your Dataset**
   * Input format must match model’s requirements.
   * Example: `[CLS] Sentence [SEP]` for BERT classification.
3. **Add Task-Specific Layer**
   * Example: linear layer for sentiment classification.
4. **Train with Low Learning Rate**
   * Freeze most layers or use **gradual unfreezing**.
   * Prevents destroying pretrained knowledge.
   * Use task-specific loss (e.g., cross-entropy).
5. **Monitor Performance**
   * Track metrics: accuracy, F1, BLEU/ROUGE.
   * Stop early if validation loss increases.
6. **Optional: Hyperparameter Tuning**
   * Batch size, learning rate, number of epochs.
7. **Deploy Fine-Tuned Model**
   * Use for inference on your task.
   * Can now **outperform generic pretrained model** on task-specific data.


---

## **4. Evaluation Metrics – BLEU Score**

**BLEU (Bilingual Evaluation Understudy)** is a metric to evaluate **machine translation quality** by comparing the model’s output to one or more reference translations.

### **How BLEU Works**
1. **N-gram Matching**
   * Compares **n-grams** (1-gram, 2-gram, etc.) in the predicted translation to the reference translation.
   * Example:
     ```
     Reference: "I love machine learning"
     Prediction: "I love learning machines"
     ```
     * 1-gram matches: I, love, learning → 3/4 = 0.75
     * 2-gram matches: I love → 1/3 ≈ 0.33
2. **Precision Score**
   * BLEU calculates **precision** for each n-gram.
3. **Brevity Penalty**
   * Penalizes translations that are **too short** compared to the reference.
4. **Weighted Average**
   * Typically, BLEU combines 1-gram to 4-gram precision with equal weights.

### **BLEU Score Range**

* **0 to 100** (or 0.0 to 1.0 in some libraries)
* **Higher = better translation quality**
* Example:
  ```
  BLEU = 0 → totally wrong translation
  BLEU = 100 → perfect match with reference
  ```
### **Why BLEU is Useful**

* Quick **automatic evaluation** of translation models.
* Works well when **multiple references** are available.
* Often used in **training loops** to track improvements in NMT (Neural Machine Translation).

### **Example Usage (Conceptual)**

Reference: "I love machine learning"
Prediction: "I enjoy machine learning"
BLEU ≈ 85 → high similarity

---

### **BLEU Score Ranges**

| BLEU Score   | Meaning                                                     |
| ------------ | ----------------------------------------------------------- |
| **0 – 10**   | Very poor translation; almost no match with reference.      |
| **10 – 30**  | Low-quality translation; some words or phrases match.       |
| **30 – 50**  | Moderate translation quality; understandable but imperfect. |
| **50 – 70**  | Good translation; captures most meaning correctly.          |
| **70 – 90**  | Very good translation; only minor errors.                   |
| **90 – 100** | Excellent translation; almost identical to reference.       |

**Notes:**

* BLEU is usually reported as **0–100** (or 0–1 if normalized).
* Scores above 50 are considered **decent for automatic evaluation**, but human judgment is still important.
* Multiple reference translations can **increase BLEU** because the model has more ways to match.





> During **evaluation** in the pipeline, BLEU is commonly used for **seq-to-seq tasks** like translation to measure how close your model’s output is to the reference text.

---



---
## 5. Key Takeaways

* Transformers = **powerful, parallelizable sequence models**.
* **Attention is everything**: shows how each token relates to others.
* **Complete NLP pipeline** = from raw data to prediction.
* Fine-tuning = **adapting pretrained models to your problem** efficiently.

---

## 6. Extra Learning Materials

### Videos (Foundations & Visual Explanations)
* [Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!! – StatQuest](https://youtu.be/zxQyTK8quyY?si=4DPOpYfwenDRP3Wu)
* **3Blue1Brown (visual learning series):**
  * [Transformers, the tech behind LLMs | Deep Learning Chapter 5](https://youtu.be/wjZofJX0v4M?si=mFCI4ddIOg8kIcjj)
  * [Attention in Transformers, step-by-step | Deep Learning Chapter 6](https://youtu.be/eMlx5fFNoYc?si=G7iNctaMfOqmc5KW)
  * [How might LLMs store facts | Deep Learning Chapter 7](https://youtu.be/9-Jl0dxWQs8?si=D5C8hUT-ttp6X_69)

### Articles & Blogs
* [How Transformers Work – Datacamp Tutorial](https://www.datacamp.com/tutorial/how-transformers-work)
* [Two minutes NLP — Learn the ROUGE metric by examples (Medium)](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499)

### Hands-on & Implementation
* [Hugging Face Transformers Course](https://huggingface.co/course/chapter1) – free coding course.
* [Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/annotated-transformer/) – build a transformer from scratch in PyTorch.

### Research Papers (Original Sources)
* [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) – the original Transformer paper.
* [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Advanced Topics & Metrics
* [Mastering Low-Rank Adaptation (LoRA) – Datacamp](https://www.datacamp.com/tutorial/mastering-low-rank-adaptation-lora-enhancing-large-language-models-for-efficient-adaptation)
* [BERTScore (arXiv)](https://arxiv.org/abs/1904.09675) – semantic evaluation metric beyond BLEU/ROUGE.
* [COMET (Facebook AI, EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.213/) – advanced MT evaluation.
---
So do you remember last task? IMDB reviews? Try to fine tune BERT base uncased and check the difference in accuracy.



