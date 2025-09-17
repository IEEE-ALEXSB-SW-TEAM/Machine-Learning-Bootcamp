# Bagging 

## I. What is Bagging?
- **Bagging (Bootstrap Aggregating)** = train the same model on many *bootstrap samples* (sampling with replacement).  
- Combine predictions → average (regression) or majority vote (classification).  
- Goal: **reduce variance** and improve stability.

### Steps
1. Draw `N` bootstrap samples (size = training set, sampled with replacement).  
2. Train a base learner (often a deep Decision Tree) on each sample.  
3. Aggregate predictions:
   - Classification → **majority vote**.  
   - Regression → **average**.

---

## II. What is Random Forest?
- A **Random Forest** = Bagging + extra randomness:
  - At each split, only a **random subset of features** is considered.
  - This decorrelates trees and usually improves performance.

---

## III. When to Use
- When data is noisy or complex.
- When the base learner has **high variance** (e.g., deep trees).
- For both **small** and **large** datasets (adjust number of trees).
- Works for:
  - **Classification** → binary or multi-class.
  - **Regression** → continuous targets.

---

## IV. Supported Tasks
| Method         | Task                       | Output            |
|----------------|---------------------------|-------------------|
| Bagging         | Classification (binary/multi) | Majority vote |
| Bagging         | Regression                | Mean of models |
| Random Forest   | Classification (binary/multi) | Majority vote |
| Random Forest   | Regression                | Mean of trees |

---

## V.  Extra Tips

### 🔍 Criterion

* Determines how the tree decides the “best” split:

  * **Classification:** `gini` (default, fast), `entropy` (information gain).
  * **Regression:** `squared_error` (minimizes variance), `mae` (less sensitive to outliers).

### ⚡ n\_jobs = -1

* Trains all trees at once using every available CPU core.
* Great for large forests → huge speedup.

### 🌲 n\_estimators

* Controls the number of trees.
* More trees → better averaging (lower variance).
* After a point, gains flatten while cost rises.

### 🎯 max\_features

* How many features to try at each split (Random Forest only).
* Smaller = more diversity between trees.
* Too small → underfit; too large → trees become similar.

### 📌 OOB Score

* Uses unused samples (“out-of-bag”) to estimate accuracy/RMSE.
* Lets you monitor performance without a validation set.


---
# Boosting
## I. What is Boosting?

Boosting is an **ensemble technique** that trains models **sequentially**, with each new model focusing on the **errors of the previous ones**.

* Goal: Reduce bias and improve accuracy.
* Unlike Bagging, models are **not independent** — each one learns from the mistakes of the previous.

**Intuition:**

> Start with a weak learner → see where it fails → train the next learner to fix these mistakes → repeat.

---

## II. How Boosting Works (General Steps)

1. Start with a **base model** (weak learner, e.g., a shallow decision tree).
2. Train it on the full dataset.
3. Evaluate errors (residuals or misclassified samples).
4. Adjust **weights**: increase importance of mispredicted samples.
5. Train a new model on this weighted data.
6. Repeat steps 3–5 for `n_estimators` rounds.
7. Combine all models’ predictions (weighted sum or vote).

![boosting](\S03-Decision Trees & Ensembles\images\boosting.jpg)



---

## III. AdaBoost (Adaptive Boosting)

AdaBoost is a classic boosting algorithm.

**Step-by-step:**

1. **Initialize weights** for all training samples equally:

   $$
   w_i = \frac{1}{N}, \quad i=1\ldots N
   $$
2. **Train a weak learner** (decision stump or small tree).
3. **Compute error** of the learner:

   $$
   \text{error}_t = \sum_i w_i \cdot I(y_i \neq \hat y_i)
   $$
4. **Compute learner weight**:

   $$
   \alpha_t = 0.5 \cdot \ln\frac{1 - \text{error}_t}{\text{error}_t}
   $$

   * Learners with smaller error get higher weight in the final prediction.
5. **Update sample weights**:

   * Increase weight for misclassified samples → they’re more important in the next round:

   $$
   w_i \leftarrow w_i \cdot e^{\alpha_t \cdot I(y_i \neq \hat y_i)}
   $$

   * Normalize weights so they sum to 1.
6. **Repeat steps 2–5** for `n_estimators` learners.
7. **Final prediction** (classification): weighted vote:

   $$
   \hat y = \text{sign}\Big(\sum_t \alpha_t \hat y_t\Big)
   $$

---
You can read more about AdaBoost [here](https://www.digitalocean.com/community/tutorials/adaboost-optimizer).  
You can check this [Medium post](https://medium.com/@sampathtunuguntla13/math-behind-adaboost-algorithm-in-3-steps-477745399553) for a mathematical example.  

> Note: Some articles (like this Medium post) describe AdaBoost with resampling, but the **standard AdaBoost algorithm does not resample**. It keeps all samples and only updates their weights after each weak learner.

If you want more details about AdaBoost, you can check [this chapter](https://www.lamda.nju.edu.cn/publication/top10chapter.pdf) from Nanjing University, China.



---

## IV. Key Points

* AdaBoost uses **simple weak learners** (decision stumps) — avoids overfitting.
* Boosting **reduces bias** by focusing on hard examples.
* More sensitive to **noisy data and outliers** than Bagging.
* Sequential training → slower than Bagging (which can be parallelized).

---

## V. Boosting vs Bagging (Comparison)

| Aspect           | Bagging                | Boosting                       |
| ---------------- | ---------------------- | ------------------------------ |
| Training         | Parallel (independent) | Sequential (each fixes errors) |
| Goal             | Reduce variance        | Reduce bias (and variance)     |
| Sample selection | Bootstrap samples      | Whole data with weights        |
| Base learner     | Usually strong         | Usually weak (stumps)          |
| Overfitting      | Less sensitive         | More sensitive                 |

---


