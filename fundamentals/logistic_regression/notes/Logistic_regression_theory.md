# Logistic Regression — Fundamentals

## 1. Problem Type

Logistic Regression is used for **binary classification problems**, where the target variable can take only two values:

- 0 (Negative class)
- 1 (Positive class)

Examples:
- Spam vs Not Spam
- Fraud vs Legitimate
- Disease vs Healthy

---

# 2. Why Linear Regression Cannot Be Used

Linear regression predicts values using:

z = wx + b

The output `z` can be **any real number**:

- -5
- 2.7
- 100

But for classification, we need a **probability**, which must lie between **0 and 1**.

Therefore, we cannot directly use linear regression for classification.

---

# 3. Predicting Probability

Instead of predicting the class directly, logistic regression predicts:

P(y = 1 | x)

Meaning:

> Probability that the output class is 1 given the input features x.

Where:

- `y` = target variable
- `x` = input features
- `P(A | B)` = probability of A **given** B (conditional probability)

Example:

P(y = 1 | x) = 0.82

This means that given the input features `x`, there is an **82% probability** that the class is `1`.

Since this is a **binary problem**:

P(y = 0 | x) = 1 - P(y = 1 | x)

---

# 4. Converting Linear Output to Probability

Our linear model still produces:

z = wx + b

This value `z` can range from **−∞ to +∞**.

We need a function that converts this value into a number between **0 and 1**.

---

# 5. Sigmoid Function

Logistic regression uses the **sigmoid function** to squash values into the range (0,1).

σ(z) = 1 / (1 + e^(-z))

Where:

z = wx + b

So the prediction becomes:

ŷ = σ(wx + b)

or

ŷ = 1 / (1 + e^-(wx + b))

---

# 6. Behavior of the Sigmoid Function

| z value | σ(z) output |
|------|------|
| Large positive | ≈ 1 |
| Large negative | ≈ 0 |
| z = 0 | 0.5 |

Meaning:

- Strong evidence for class 1 → probability close to 1
- Strong evidence for class 0 → probability close to 0

---

# 7. Converting Probability to Class

Once probability is predicted:

If:

ŷ ≥ 0.5 → predict class **1**

Else:

ŷ < 0.5 → predict class **0**

This threshold can be adjusted depending on the application.

---

# 8. Loss Function (Why Not MSE)

Mean Squared Error (MSE) is not ideal for logistic regression because the sigmoid function makes the optimization landscape difficult.

Instead, we use **Binary Cross Entropy (Log Loss)**.

---

# 9. Binary Cross Entropy Loss

For a single data point:

L = - [ y log(ŷ) + (1 - y) log(1 - ŷ) ]

Where:

- `y` = true label
- `ŷ` = predicted probability

---

## Case 1: True label = 1

Loss becomes:

L = -log(ŷ)

| Prediction | Loss |
|---|---|
| 0.99 | Very small |
| 0.1 | Very large |

---

## Case 2: True label = 0

Loss becomes:

L = -log(1 - ŷ)

| Prediction | Loss |
|---|---|
| 0.1 | Very small |
| 0.9 | Very large |

---

# Key Intuition

Log loss **heavily penalizes confident but incorrect predictions**, which helps the model learn better probabilities.

---

# Final Logistic Regression Model

1. Compute linear score

z = wx + b

2. Convert to probability

ŷ = 1 / (1 + e^(-z))

3. Optimize parameters using **Binary Cross Entropy Loss**.