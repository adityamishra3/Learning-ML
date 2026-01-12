# 📘 All Maths Required for Linear Regression (Basics)


## Step 1. Model (Hypothesis) Formula

### Single feature

$
\hat{y} = wx + b
$

### Multiple features

$
\hat{y} = w_1x_1 + w_2x_2 + \dots + w_nx_n + b
$

Where:

* $\hat{y} $ → predicted output
* $ w, w_1, \dots, w_n $ → weights
* $ b $ → bias (intercept)

---

## Step 2. Dataset

Assume we have a dataset with **m rows** and **one feature**.

| x  | y  |
| -- | -- |
| 10 | 20 |
| 12 | 23 |
| 14 | 28 |

Where:

* $ x $ = input feature
* $ y $ = actual output

---

## Step 3. Prediction for Each Data Point

For the $ i^{th} $ data point:

$
\hat{y}_i = wx_i + b
$

---

## Step 4. Error (Residual)

Error for one data point:

* $
\text{error}_i = \hat{y}_i - y_i
$

This tells **how wrong** the prediction is.

---

## Step 5. Cost Function (Mean Squared Error)

To measure how good or bad our model is overall, we use a **cost function**.

### Mean Squared Error (MSE)

$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$

Substitute $ \hat{y}_i = wx_i + b $:

$
J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} (wx_i + b - y_i)^2
$

Where:

* $ m $ = number of data points
* $ \frac{1}{2} $ is used for mathematical convenience during differentiation

Lower cost ⇒ better ( w ) and ( b )

---

## Step 6. Objective of Training

The goal of linear regression training is:

$
\min_{w,b} J(w,b)
$

That is, find values of ( w ) and ( b ) that **minimize the cost function**.

---

## Step 7. Gradient Descent (Optimization Algorithm)

Gradient descent is used to **iteratively update** parameters to reduce cost.

### Core update rule

$
\theta_{new} = \theta_{old} - \alpha \frac{\partial J}{\partial \theta}
$

Where:

* $ \theta $ is a parameter $( w ) or ( b )$
* $ \alpha $ is the learning rate

---

## Step 8. Partial Derivatives of the Cost Function

### Partial derivative w.r.t. weight ( w )

$
\frac{\partial J}{\partial w}
= \frac{1}{m} \sum_{i=1}^{m} ( \hat{y}_i - y_i ) x_i
$

---

### Partial derivative w.r.t. bias ( b )

$
\frac{\partial J}{\partial b}
= \frac{1}{m} \sum_{i=1}^{m} ( \hat{y}_i - y_i )
$

---

## Step 9. Gradient Descent Update Equations (Batch GD)

Using the derivatives above:

### Weight update

$
w := w - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} ( \hat{y}_i - y_i ) x_i
$

### Bias update

$
b := b - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} ( \hat{y}_i - y_i )
$

This update uses **all m data points** → called **Batch Gradient Descent**.

---

## Step 10. Stochastic Gradient Descent (SGD)

Instead of using all data points, SGD updates parameters using **one data point at a time**.

### Loss for one data point

$
J^{(i)} = \frac{1}{2}(\hat{y}_i - y_i)^2
$

### Updates

$
w := w - \alpha (\hat{y}_i - y_i)x_i
$

$
b := b - \alpha (\hat{y}_i - y_i)
$

SGD is faster but noisier.

---

## Step 11. Mini-Batch Gradient Descent

Uses a batch of ( k ) data points.

### Cost for mini-batch
$
J = \frac{1}{2k} \sum_{j=1}^{k} (\hat{y}_j - y_j)^2
$

### Updates

$
w := w - \alpha \cdot \frac{1}{k} \sum_{j=1}^{k} ( \hat{y}_j - y_j ) x_j
$

$
b := b - \alpha \cdot \frac{1}{k} \sum_{j=1}^{k} ( \hat{y}_j - y_j )
$

This is the **most commonly used method in practice**.

---
### Any one of the gradient descent is used, depending on the usecase and dataset.
---

## Step 12. Epochs

* One **epoch** = one full pass over the training dataset
* Multiple epochs allow the model to gradually improve ( w ) and ( b )

---

## Step 13. Final Trained Model

After training:

$
\boxed{\hat{y} = wx + b}
$

This equation is:

* saved to disk
* used to predict unseen data

---

## Step 14. Key Properties of Linear Regression

* MSE cost surface is **convex**
* Only **one global minimum**
* Gradient descent will converge (with proper learning rate)

---

## Final Summary (Mental Model)

* Model → linear equation
* Error → prediction − actual
* Cost → average squared error
* Gradient → direction to reduce error
* Learning rate → step size
* Epoch → repeated refinement

---
