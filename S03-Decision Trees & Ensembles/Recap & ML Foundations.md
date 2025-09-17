# Recap & ML Foundations 🔁💡


## 📚 Regression vs Classification

| Task Type      | Output              | Example                              | Models Used                    |
|----------------|---------------------|--------------------------------------|--------------------------------|
| Regression     | Continuous number   | Predict house price                  | Linear Regression              |
| Classification | Categorical label   | Predict if email is spam or not      | Logistic Regression, KNN, Trees|

---

## 🧠 Model Recap

### 📈 Linear Regression

Used for regression problems. It tries to fit a straight line to data:

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

Goal: minimize Mean Squared Error (MSE)

---

### 📉 Logistic Regression

Used for classification. It outputs **probability** of a class using a sigmoid:

$$
P(y=1 \mid x) = \frac{1}{1 + e^{-(w^T x)}}
$$

- Output between 0 and 1
- Threshold at 0.5 to decide the class
- Trained using cross-entropy loss
---
## NEW: Softmax in Logistic Regression

### Introduction

In Logistic Regression, when we have more than two classes (i.e., a multi-class classification problem), we need to extend the basic binary logistic regression model. The **Softmax function** is commonly used in such cases, especially in **multinomial logistic regression**.

The **Softmax function** takes a vector of raw scores (often called **logits**) and converts them into probabilities by normalizing the output values. Each probability corresponds to the likelihood of a given input belonging to a specific class.

### Formula

Given a vector of raw scores (logits) $z = [z_1, z_2, ..., z_K]$, the Softmax function computes the probability of class $( i )$ as follows:

$$
P(y = i | x) = \frac{e^{z_i}}{\sum_{k=1}^{K} e^{z_k}}
$$

Where:
- $P(y = i | x) )$ is the probability that input $( x )$ belongs to class $( i )$.
- $( z_i )$ is the raw score (logit) for class $( i )$.
- $( K )$ is the total number of classes.
- The denominator $( \sum_{k=1}^{K} e^{z_k} )$ is the sum of the exponentials of all logits, ensuring that the sum of all class probabilities equals 1.

### How it Works

1. **Raw Scores (Logits)**: In a classification problem, we typically have raw outputs (logits) from a model, such as a neural network, that correspond to each class.
   
2. **Exponentiation**: Softmax exponentiates each of these raw scores, making them positive and larger for higher logits.

3. **Normalization**: The sum of all these exponentiated scores is computed and used to normalize the individual values. This step ensures that the probabilities add up to 1, which is a requirement for probability distributions.

4. **Interpretation**: After applying Softmax, the output for each class represents the probability that the input \( x \) belongs to that class. The class with the highest probability is chosen as the model’s prediction.

### Use Case in Logistic Regression

In binary logistic regression, we model the probability of one class (e.g., class 1) using the sigmoid function. In multinomial logistic regression (for multi-class classification), we use the Softmax function instead. Each class gets a probability score, and we can select the class with the highest probability as the final prediction.

#### Example:

Suppose we have a three-class problem, with raw scores (logits) for a given input \( x \) as follows:

$$
z = [2.0, 1.0, 0.1]
$$

To compute the probabilities, we apply the Softmax function:

$$
P(y = 1 | x) = \frac{e^{2.0}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.659
$$
$$
P(y = 2 | x) = \frac{e^{1.0}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.242
$$
$$
P(y = 3 | x) = \frac{e^{0.1}}{e^{2.0} + e^{1.0} + e^{0.1}} \approx 0.099
$$

Thus, the model predicts that the input belongs to class 1, as it has the highest probability (0.659).

### Conclusion

Softmax is a powerful tool for multi-class classification, providing a way to model probabilities for each class. It ensures that the model's output can be interpreted as a probability distribution over the classes, making it useful in tasks like image classification, text categorization, and more.



---

## NEW: Multiclass Cross-Entropy

**Used for:** Classification tasks with more than two classes (multi-class classification).

---

### **1. Concept**

Multiclass cross-entropy measures how well a model’s predicted probability distribution **matches the true class**.

* True labels are usually **one-hot encoded**.
* Predictions come from a **softmax layer**, giving probabilities for each class.

> It penalizes the model when it assigns low probability to the correct class.

---

### **2. Formula**

$$
\text{Loss} = - \sum_{i=1}^{C} y_i \cdot \log(\hat{y}_i)
$$

Where:

* $C$ = number of classes
* $y_i$ = 1 if the true class is $i$, otherwise 0
* $\hat{y}_i$ = predicted probability for class $i$

> Because $y_i$ is one-hot, only the correct class contributes to the sum.

$$
\text{Loss} = -\log(\hat{y}_{\text{true class}})
$$

---

### **3. Intuition**

* Correct class probability near **1** → loss ≈ 0
* Correct class probability near **0** → loss is **very high**
* Encourages the model to **assign high probability to the correct class**.

---

### **4. Example**

True class: 3 (out of 4 classes) → one-hot: `[0, 0, 1, 0]`
Model prediction: `[0.1, 0.2, 0.6, 0.1]`

$$
\text{Loss} = -\log(0.6) \approx 0.51
$$

Model predicts `[0.1, 0.2, 0.1, 0.6]` → Loss = `-log(0.1) ≈ 2.3` → much worse

---

### **5. Key Notes**

* Usually combined with **softmax activation** in the output layer.
* Works for **any number of classes**.
* Penalizes **confident wrong predictions** heavily.

---

---

## 📍 K-Nearest Neighbors (KNN)

- **Lazy learner** (no training phase)
- To predict a label: look at the labels of the K closest points
- Distance metric: usually Euclidean

#### Pros:
- Simple, effective
- No training time

#### Cons:
- Slow with large datasets
- Sensitive to irrelevant features

---

## 🧪 A Typical ML Pipeline

1. **Understand the problem**  
   → Classification or regression?

2. **Collect & clean data**  
   → Handle missing values, duplicates

3. **Preprocessing**  
   - Normalize/standardize data
   - Encode categorical variables
   - Feature selection

4. **Split data**  
   → Train / Test (e.g., 80/20 split)

5. **Train a model**  
   → Use scikit-learn, PyTorch, etc.
   Note: for learning purposes, you can try make the model yourself.

6. **Evaluate performance**
   - Accuracy (classification)
   - MSE/R² (regression)

7. **Optimize model**  
   - Tune hyperparameters
   - Try better features or different models

8. **Deploy**  
   → Build an app, API, or dashboard

---

## Evaluation Metrics 🧪🎯

When you're building a classification model, accuracy is **not always enough**, especially with **imbalanced datasets** (e.g., 95% negative, 5% positive).

That's why we use more detailed metrics:  

**Precision, Recall, and F1 Score** 🔍

---

### 📦 Confusion Matrix

A table that summarizes how well a classifier performs:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)     | False Negative (FN)    |
| **Actual Negative** | False Positive (FP)    | True Negative (TN)     |

---

### 🎯 Accuracy

The simplest metric:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

> **When is it useful?**  
Useful when **classes are balanced** and **all errors are equally bad**.

> **Example**  
In digit recognition (MNIST), where each class is balanced, accuracy is a good measure.

---

### 📌 Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

How many of the **positive predictions** were **actually correct**?

> **When is it important?**  
When **false positives** are costly.

> **Example**  
-> **Email Spam Detection**:  
  Precision matters — if the model wrongly marks a real email as spam (false positive), the user might miss something important.
-> **Cheating Detection**:  
  Precision matters — if the model wrongly marks a student as cheater (false positive), the student will be wronged.

---

### 🔍 Recall (Sensitivity / True Positive Rate)

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

How many of the **actual positives** did the model **correctly find**?

> **When is it important**
> When **false negatives** are costly.

> **Example**  
-> **Cancer Detection**:  
  Recall matters — we don’t want to miss a real cancer case (false negative), even if it means getting some false alarms.

---

### ⚖️ F1 Score

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

A **harmonic mean** of precision and recall. Balances both.

> **When is it useful?**  
When you want to balance precision and recall, especially on **imbalanced datasets**.

> **Example**  
-> **Fraud Detection**:  
  You care about both catching fraud (recall) and minimizing false alarms (precision), so F1 is perfect.

---
### 🧪 Real-World Summary

| Task                      | Priority Metric | Why? |
|---------------------------|------------------|------|
| Email Spam Filter         | Precision        | Don't mark important emails as spam |
| Cancer Diagnosis          | Recall           | Don’t miss a real cancer case |
| Face Recognition Login    | Precision        | Don’t allow unauthorized access |
| Fraud Detection           | F1 Score         | Balance catching fraud & avoiding false alarms |
| Balanced Dataset (e.g. MNIST) | Accuracy       | All classes matter equally |



---

## NEW: R² Score (Coefficient of Determination)

**Used for:** Regression problems (predicting continuous numeric values).

---

### 1. Formula

$$
R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
$$

Where:

* **Residual Sum of Squares (SS$_\text{res}$)**:

$$
\text{SS}_{\text{res}} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Measures **how far predictions are from actual values**.

* **Total Sum of Squares (SS$_\text{tot}$)**:

$$
\text{SS}_{\text{tot}} = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

Measures **how far actual values are from their mean**.

---

### **2. Intuition**

* $R^2 = 1$ → perfect predictions ($y_i = \hat{y}_i$ for all i)
* $R^2 = 0$ → model predicts **mean of y** for all points
* $R^2 < 0$ → model is worse than just predicting the mean

Think: **“What fraction of variance in y is explained by the model?”**

---

### **3. Example**

Actual values: $y = [3, 5, 4, 7]$
Predicted: $\hat{y} = [2.8, 4.9, 4.1, 6.8]$

1. Calculate $\bar{y} = 4.75$
2. SS$_\text{tot} = (3-4.75)^2 + (5-4.75)^2 + ... = 8.75$
3. SS$_\text{res} = (3-2.8)^2 + (5-4.9)^2 + ... = 0.1$
4. $R^2 = 1 - 0.1 / 8.75 \approx 0.988$ → very good fit

---

### **4. Pros and Cons**

**Pros:**

* Easy to interpret
* Standard metric for regression

**Cons:**

* Can be negative if the model is bad
* Doesn’t tell you **absolute error magnitude** (use RMSE/MAE too)

---

## NEW: ROC and AUC 

### 1. What is ROC?

**ROC** stands for **Receiver Operating Characteristic** curve. It is a graphical plot that illustrates the **performance of a binary classifier** as its **decision threshold** is varied.

* **Binary classifier**: A model predicting `0` (negative) or `1` (positive).
* **Threshold**: The probability value above which we predict class `1`. Default is usually `0.5`.

---

### 2. Key Terms

To understand ROC, we need:

| Term                    | Meaning                       |
| ----------------------- | ----------------------------- |
| **TP (True Positive)**  | Model predicts 1, actual is 1 |
| **TN (True Negative)**  | Model predicts 0, actual is 0 |
| **FP (False Positive)** | Model predicts 1, actual is 0 |
| **FN (False Negative)** | Model predicts 0, actual is 1 |

From this, we calculate:

1. **True Positive Rate (TPR) / Sensitivity / Recall**
   Measures the proportion of actual positives correctly identified.

$$
\text{TPR} = \frac{TP}{TP + FN}
$$

2. **False Positive Rate (FPR)**
   Measures the proportion of actual negatives incorrectly classified as positive.

$$
\text{FPR} = \frac{FP}{FP + TN}
$$

---

### 3. How ROC Curve is Drawn

* **X-axis**: FPR (False Positive Rate)
* **Y-axis**: TPR (True Positive Rate)

**Steps**:

1. Sort all predicted probabilities from **highest to lowest**.
2. Start with threshold = ∞ → predict all negatives → TPR=0, FPR=0.
3. Move threshold down gradually → predict more positives → calculate TPR and FPR.
4. Plot `(FPR, TPR)` points → connect to get ROC curve.

> Intuition: The curve shows **trade-off** between detecting positives (TPR) vs mistakenly labeling negatives as positive (FPR).

---

### 4. Example

Suppose we have 5 samples with actual and predicted probabilities:

| Actual | Predicted Prob |
| ------ | -------------- |
| 1      | 0.9            |
| 1      | 0.8            |
| 0      | 0.7            |
| 0      | 0.4            |
| 1      | 0.3            |

* Threshold = 0.5 → predict positives if prob ≥ 0.5

  * Predicted: `[1,1,1,0,0]`
  * TPR = 2/3 ≈ 0.67
  * FPR = 1/2 = 0.5
* Threshold = 0.7 → predict `[1,1,0,0,0]`

  * TPR = 2/3 ≈ 0.67
  * FPR = 0/2 = 0

Plot these points to get ROC curve.

---

### 5. What is AUC?

**AUC** = **Area Under the ROC Curve**.

* Ranges **0 to 1**
* Measures **classifier’s ability to distinguish between positive and negative classes**.
* Interpretation:

  * **0.5** → random guessing
  * **>0.7** → acceptable
  * **>0.8** → good
  * **>0.9** → excellent

> Geometric interpretation: probability that a randomly chosen positive sample has a higher score than a randomly chosen negative sample.

---

### 6. Why Only Positive Class Matters

* ROC focuses on **how well you separate positives from negatives**, not absolute probability correctness.
* **TPR is about positives**.
* **FPR is about negatives misclassified as positives**.
* Together, the ROC captures the classifier’s discrimination ability.

---

### 7. Threshold Movement

* Changing the threshold moves along the ROC curve:

  * **High threshold** → fewer positives predicted → low TPR, low FPR
  * **Low threshold** → more positives predicted → high TPR, high FPR

> ROC curve is **threshold-independent** summary of all possible thresholds.

---

### 8. Summary

* **ROC Curve**: Shows trade-off between TPR and FPR for all thresholds.
* **AUC**: Measures **overall classifier performance**.
* **Good classifier** → ROC curve bows towards top-left → AUC close to 1.
* **Random classifier** → diagonal line → AUC ≈ 0.5.

---

You can read about it more on [this page](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).


✅ Now that we’ve made our recap, let’s dive into **Decision Trees** — one of the most interpretable ML models.