# Session 1: Introduction to Machine Learning & Linear Regression

Welcome to the first session of the Machine Learning Bootcamp! In this session, we’ll cover the basics of machine learning, key concepts, and dive into our first algorithm: **Linear Regression**. By the end of this session, you’ll have a solid understanding of the fundamentals and hands-on experience building your first ML model.

---

## Agenda
1. **What is Machine Learning?**
2. **Types of Machine Learning**
3. **Key Concepts: Features, Labels, Training, and Testing**
4. **Overfitting and Underfitting**
5. **Introduction to Linear Regression**
6. **Workshop: Building a Linear Regression Model**

---

## 1. Artificial Intelligence, Machine Learning, and Deep Learning

### 1. What is Artificial Intelligence (AI)?
Artificial Intelligence (AI) is a broad field of computer science focused on creating systems that can perform tasks requiring human-like intelligence (reasoning, decision-making, problem-solving, language understanding, etc.).  

AI is the **umbrella concept** that includes both **Machine Learning (ML)** and **Deep Learning (DL)**, but it also covers older symbolic and rule-based approaches.  

#### 🔹 AI Examples That Are *Not* ML/DL
- **Rule-based expert systems** → medical diagnosis using “if–then” rules.  
- **Classic game-playing AI** → IBM’s *Deep Blue* (1997) chess program (used search, not learning).  
- **Search algorithms** → A* algorithm in GPS or video games.  
- **Logic-based systems** → Prolog programs solving puzzles using reasoning.  

---

### 2. What is Machine Learning (ML)?
Machine Learning (ML) is a **subset of AI** where systems learn patterns from data and improve performance over time without being explicitly programmed.  
Instead of writing rules, we train models to find relationships in data.

#### 🔹 ML Examples
- **Predicting house prices** → based on size, location, and number of bedrooms.  
- **Spam classification** → detecting spam vs. non-spam emails.  
- **Product recommendations** → e-commerce suggesting “you may also like.”  

---

### 3. What is Deep Learning (DL)?
Deep Learning (DL) is a **specialized subset of ML** that uses **artificial neural networks with many layers** (“deep” networks).  
It is especially powerful for handling large, unstructured data such as images, speech, and text.

#### 🔹 DL Examples
- **Image recognition** → detecting objects or faces in photos.  
- **Natural Language Processing (NLP)** → ChatGPT, Google Translate.  
- **Speech recognition** → converting voice to text (e.g., Google Voice Typing).  
- **Autonomous vehicles** → detecting pedestrians, road signs, and making driving decisions.  

---

### 🔗 Relationship (AI → ML → DL)
- **AI** → The big umbrella (all intelligent systems).  
  - **ML** → AI that learns from data.  
    - **DL** → ML using multi-layer neural networks.  

![AI-vs-ML-vs-Deep-Learning](/S1-Intro_to_ML/images/AI-vs-ML-vs-Deep-Learning.png)


This bootcamp focuses mainly on **Machine Learning (ML)** and **Deep Learning (DL)**.  
That means we will dive into how systems **learn from data (ML)** and how **neural networks (DL)** can handle complex tasks like image recognition, and natural language processing.

---

## 2. Types of Machine Learning
There are three main types of ML:

### a) Supervised Learning
- The model learns from **labeled data** (input-output pairs).
- Examples: Linear Regression, Logistic Regression, Decision Trees.

### b) Unsupervised Learning
- The model learns from **unlabeled data** and finds hidden patterns.
- Examples: Clustering (K-Means), Dimensionality Reduction (PCA).

### c) Reinforcement Learning
- The model learns by interacting with an environment and receiving rewards or penalties.
- Examples: Game-playing AI (e.g., AlphaGo), robotics.

Since we will start this bootcamp with **Supervised Learning**, it’s important to understand its types.  
In supervised learning, the model is trained on **labeled data** (we already know the correct output during training).  

### 🔹 Types of Supervised Learning

1. **Classification**  
   - **Definition**: Predicts a **category (discrete label)**.  
   - **Examples**:  
     - Spam vs. Not Spam (email filtering)  
     - Cat vs. Dog (image recognition)  
     - Disease Positive vs. Negative (medical diagnosis)  

2. **Regression**  
   - **Definition**: Predicts a **continuous numerical value**.  
   - **Examples**:  
     - Predicting house prices  
     - Estimating a person’s salary based on experience  
     - Forecasting stock prices  

**In short**:  
- **Classification** → Answers *“Which class does it belong to?”*  
- **Regression** → Answers *“What is the numeric value?”*  

![regression-vs-classification_simple-comparison-image_v3](/S1-Intro_to_ML/images/regression_vs_classification.png)


---

## 3. Key Concepts
### Features (Input Variables)
- The attributes used to make predictions (e.g., size, location for house price prediction).
- Features are either continuous (e.g., price) or categorical (type of sport he is playing)

### Labels (Output Variables)
- The target variable we want to predict (e.g., house price).
- Multiclass, means that each sample belongs to one category (e.g., dog/cat/bird), mutlilabel means (cat AND dog).

### Training and Testing
- **Training Data**: Used to teach the model.
- **Testing Data**: Used to evaluate the model’s performance on unseen data.

### Validation
- A technique to ensure the model generalizes well to new data (e.g., cross-validation).

---

## 4. Overfitting and Underfitting
### Overfitting
- The model learns the training data too well, including noise, and performs poorly on new data.
- **Solution**: Simplify the model, use regularization, or get more data.

### Underfitting
- The model is too simple to capture the underlying patterns in the data.
- **Solution**: Use a more complex model or add more features.

![mlconcepts_image5](/S1-Intro_to_ML/images/balanced.png)


### Bias-variance tradeoff
When building ML models, there is a fundamental tradeoff between **bias** and **variance**:

- **Bias**: Error caused by oversimplifying the model (e.g., underfitting).  
- **Variance**: Error caused by the model being too sensitive to training data (e.g., overfitting).  
- **Goal**: Find a balance where the model is complex enough to capture patterns but simple enough to generalize well.

### Visualization
- **High Bias (Underfitting)** → Model is too simple, misses important patterns.  
- **High Variance (Overfitting)** → Model memorizes training data, fails on new data.  
- **Good Tradeoff** → Balanced model that performs well on both training and unseen data.

![r7QFy](/S1-Intro_to_ML/images/dart_bias_variance.png)

![1_oO0KYF7Z84nePqfsJ9E0WQ](/S1-Intro_to_ML/images/bias_variance_tradeoff.png)

### Bias-Variance Tradeoff

#### Physical Meaning

##### Bias (Systematic Error)
Think of **bias** as how far your model’s average prediction is from the true value.  

- **High bias** = your model is too simple (**underfitting**).  
- It makes strong assumptions and misses patterns.  

**In the graph**: as model complexity increases, bias decreases (because the model starts fitting the data better).  



---

##### Variance (Sensitivity to Data)
Variance measures how much your model’s predictions change if you train it on a different dataset.  

- **High variance** = your model is too complex (**overfitting**).  
- It memorizes noise instead of learning general patterns.  

**In the graph**: as model complexity increases, variance goes up (because the model adapts too much to the training set).  


---

##### Irreducible Error
Even the best possible model cannot predict perfectly because of random noise in the data.  

Examples of irreducible error:  
- Measurement errors (e.g., faulty sensor readings).  
- Natural randomness (e.g., stock price fluctuations).  
- Missing features (e.g., predicting house price without knowing the owner’s negotiation skills).  

No matter how complex the model is, there’s always a **baseline error** we cannot reduce, that’s the *irreducible error*.  

---

#### Total Error Formula
In supervised learning, the total error can be expressed as:  

**Total Error = Bias² + Variance + Irreducible Error**  

This explains the U-shaped curve often shown in the bias-variance tradeoff graph:  
- At low complexity → high bias dominates.  
- At high complexity → high variance dominates.  
- In the middle → best tradeoff.  

Note: In regression, the expected Mean Squared Error (MSE) can be decomposed into Bias² + Variance + Irreducible Error.

---

#### The Four Cases

| Bias  | Variance | Case          | Example                          |
|-------|----------|---------------|----------------------------------|
| High  | Low      | Underfitting  | Linear model on complex data     |
| Low   | High     | Overfitting   | Deep decision tree memorizing noise |
| Low   | Low      | Ideal         | Balanced model (sweet spot)      |
| High  | High     | Worst case    | Wrong model + bad data           |


---

## 5. Introduction to Linear Regression

### Overview
Linear Regression is a **supervised learning algorithm** used to model the relationship between a dependent variable (output) and one or more independent variables (inputs). It is primarily used for **predicting continuous values**, making it one of the fundamental techniques in machine learning and statistics.

---

### 1. Mathematical Forms
#### Equation of line
The simplest form of linear regression is the equation of a straight line:

$y = mx + b$

Where:
- $y$ is the dependent variable (output)
- $x$ is the independent variable (input feature)
- $m$ is the slope (weight) of the line, indicating how much $y$ changes with respect to $x$
- $b$ is the intercept (bias), representing the value of $y$ when $x = 0$
#### Vector / Matrix Form of Linear Regression

For one sample:
$$y = x w + b$$

Where:

- \(x\) is the **feature vector** of shape \((1, d)\)  
- \(w\) is the **weight vector** of shape \((d, 1)\)  
- \(b\) is the scalar bias  
- \(y\) is the scalar prediction

---

For all \(n\) samples:
$$
\mathbf{y} = X w + b
$$

Where:

- $X \in \mathbb{R}^{n \times d}$ is the **design matrix** (each row is a sample)  
- $w \in \mathbb{R}^{d \times 1}$ is the weight vector  
- $b$ can be treated as a scalar or as a vector of shape $(n, 1)$ if broadcast  
- $\mathbf{y} \in \mathbb{R}^{n \times 1}$ is the vector of predictions

##### Linear Regression as Matrix Multiplication

Linear regression can be expressed simply as:

$$
\mathbf{y} = X w
$$

Where:

- $X \in \mathbb{R}^{n \times (d+1)}$ — the **design matrix** (each row is a sample, each column is a feature)  
- $w \in \mathbb{R}^{(d+1) \times 1}$ — the vector of weights (parameters)  
- $\mathbf{y} \in \mathbb{R}^{n \times 1}$ — the predictions for all $n$ samples

By adding a column of ones to $X$, the first weight $w_0$ acts as the bias.

#### Polynomial Regression
Linear regression can be extended to capture **non-linear** relationships using polynomial terms. This is called **Polynomial Regression**:


$y = a_0 + a_1x + a_2x^2 + a_3x^3 + \dots + a_nx^n$

- There can be multiple features $x_1, x_2, \dots, x_n$, each representing different input variables.
- Each feature can be raised to different powers to model complex relationships.
- If $n = 1$, it's a simple linear regression.
- If $n = 2$, it's a quadratic regression.
- If $n = 3$, it's a cubic regression, and so on.

This allows linear regression to model more complex relationships by **transforming features** before applying linear regression techniques.

#### Exponential Transformations
Linear regression can also model exponential relationships by applying a logarithmic transformation to the dependent variable. For example, if the relationship is of the form:

$y = e^{a_0 + a_1x}$

Taking the natural logarithm of both sides transforms it into a linear form:

$ln(y) = a_0 + a_1x$

This can now be solved using standard linear regression techniques.

---

### 2. Cost Function: Mean Squared Error (MSE)
A key aspect of training a regression model is **quantifying the error** between predicted values and actual values. This is done using a **cost function**.

The most commonly used cost function for linear regression is **Mean Squared Error (MSE)**:

$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

Where:
- $y_i$ is the actual output.
- $\hat{y}_i$ is the predicted output.
- $n$ is the number of data points.

#### Why Use MSE Instead of Absolute Error?
1. **Differentiability**: MSE is differentiable, making it easier to optimize using **Gradient Descent**.
2. **Emphasizes Large Errors**: Squaring errors penalizes larger errors more than smaller ones, improving sensitivity to significant mistakes.
3. **Smooth Optimization**: The squared function provides a smooth gradient, leading to stable convergence.

---

### 3. Optimization Methods

#### 1. Closed-Form Solution (Normal Equation)
Before using iterative approaches like gradient descent, we can compute the optimal parameters directly using the **Normal Equation**:

$w = (X^T X)^{-1} X^T y$

Where:
- $X$ is the feature matrix.
- $y$ is the target vector.
- $w$ is the weight vector.

This method avoids iterative updates but can be computationally expensive for large datasets, in addition to that, $(X^T X)$ isn't always invertible.

#### 2. Gradient Descent
**Gradient Descent** is an optimization algorithm used to minimize the cost function by iteratively adjusting the model’s parameters (weights and bias).

##### How It Works:
1. **Compute the Gradient**: The gradient of the cost function with respect to model parameters is computed.
2. **Update Parameters**: Move in the direction of the negative gradient to reduce cost.
3. **Repeat Until Convergence**: The process continues until the changes become minimal or reach a stopping criterion.

##### Formal Update Rules:
For parameters $w_j$ (weights) and $b$ (bias), the updates follow:

$w_j := w_j - \alpha \cdot \frac{\partial J(w, b)}{\partial w_j}$
$b := b - \alpha \cdot \frac{\partial J(w, b)}{\partial b}$

Where:
- $J(w, b)$ is the cost function (MSE).
- $\alpha$ is the **learning rate**, controlling the step size.
- The partial derivatives $\frac{\partial J}{\partial w_j}$ and $\frac{\partial J}{\partial b}$ guide the direction of parameter updates.

##### Types of Gradient Descent
###### 1. **Batch Gradient Descent**
- Computes the gradient using the **entire dataset**.
- More stable but computationally expensive for large datasets.
- Slower convergence.

###### 2. **Stochastic Gradient Descent (SGD)**
- Updates parameters using **one data point** at a time.
- Computationally efficient but introduces more noise.
- Faster convergence but fluctuates around the minimum.

###### 3. **Mini-Batch Gradient Descent**
- A compromise between batch and stochastic gradient descent.
- Uses **a small batch of data points** per update.
- Reduces variance while maintaining computational efficiency.

---

### 4. Regularization: L1 and L2
Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. The two most common types of regularization are **L1 (Lasso)** and **L2 (Ridge)**.

#### L1 Regularization (Lasso)
L1 regularization adds the absolute value of the weights to the cost function:

$J(w, b) = \text{MSE} + \lambda \sum_{j=1}^{n} |w_j|$

Where:
- $\lambda$ is the regularization parameter, controlling the strength of the penalty.
- L1 regularization encourages sparsity, meaning it can shrink some weights to zero, effectively performing feature selection.

#### L2 Regularization (Ridge)
L2 regularization adds the squared magnitude of the weights to the cost function:

$J(w, b) = \text{MSE} + \lambda \sum_{j=1}^{n} w_j^2$

Where:
- $\lambda$ is the regularization parameter.
- L2 regularization discourages large weights but does not force them to zero, leading to smoother models.

#### Elastic Net
Elastic Net combines L1 and L2 regularization:


$J(w, b) = \text{MSE} + \lambda_1 \sum_{j=1}^{n} |w_j| + \lambda_2 \sum_{j=1}^{n} w_j^2$

This provides a balance between feature selection (L1) and weight shrinkage (L2).

---

## 6. Making Your Own Model
Now that you understand linear regression, you can start building your own model using Python with libraries like `scikit-learn`. The key steps include:

1. **Collecting Data** – Gather relevant datasets that contain input and output variables.
2. **Preprocessing Data** – Handle missing values, remove outliers, normalize data, and ensure that features are properly formatted.
3. **Splitting Data** – Divide the dataset into training and testing sets to evaluate performance.
4. **Choosing the Right Model** – Decide whether to use simple linear regression or polynomial regression based on data patterns.
5. **Training the Model** – Fit the model to the training data using optimization techniques like gradient descent.
6. **Evaluating Performance** – Use performance metrics such as MSE and R² Score to assess model accuracy.
7. **Making Predictions and Fine-Tuning** – Adjust hyperparameters, apply feature selection, and optimize regularization to improve results.

### Fine-Tuning Your Model
To improve your model’s performance, consider:
1. **Feature Selection and Engineering**
   - Remove irrelevant features using correlation analysis or feature importance.
   - Apply polynomial transformation, normalization, or log transformations.
   - Check for multicollinearity using Variance Inflation Factor (VIF).

2. **Hyperparameter Tuning**
   - Apply **Ridge (L2) or Lasso (L1) regularization** to prevent overfitting.
   - Adjust the **learning rate** for better convergence.
   - Choose the best **polynomial degree** to balance bias and variance.

3. **Cross-Validation**
   - Use **k-fold cross-validation** to evaluate the model on multiple data splits.
   - Helps in selecting the best model by preventing overfitting.


Fine-tuning ensures that your model generalizes well to new data and provides more accurate predictions.

---

### **Frequently Asked Questions (FAQ)**  

**Q: Why is it important to understand the theoretical background of Machine Learning if pre-built libraries handle most implementations?**  

**A:** Understanding the theoretical foundations of Machine Learning (ML) is essential for several reasons:  

1. **Model Interpretation** – Theoretical knowledge helps in understanding how models work, why they produce certain results, and how to interpret their predictions.  

2. **Hyperparameter Optimization** – Many ML algorithms require fine-tuning of hyperparameters. A solid theoretical background enables informed decision-making rather than relying on trial and error.  

3. **Avoiding Common Pitfalls** – Concepts such as the bias-variance tradeoff, overfitting, and underfitting are crucial for developing models that generalize well to new data.  

4. **Selecting the Appropriate Model** – Different ML problems require different approaches. Understanding the strengths and limitations of various models ensures the selection of the most suitable one.  

5. **Performance Optimization** – Knowledge of mathematical principles such as gradient descent, loss functions, and optimization techniques allows for performance improvements and debugging of training issues.  

6. **Customization and Development** – While libraries provide pre-built solutions, real-world applications often require modifications. Understanding ML theory allows for the customization and development of new models or enhancements.  

7. **Result Interpretation and Compliance** – In fields like healthcare, finance, and autonomous systems, it is crucial to explain model decisions for trust, compliance, and regulatory purposes.  

8. **Advancement in ML** – For those aiming to contribute to research, develop new algorithms, or optimize existing ones, theoretical knowledge is indispensable.  

Even though ML libraries simplify implementation, a strong theoretical foundation enhances problem-solving abilities, ensures responsible model deployment, and allows for innovation in the field.

---