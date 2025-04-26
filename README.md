# pricePrediction
**Used Linear Regression Model to predict the Price of the house by analyzing the data set**

### Why Linear Regression?

- **Natural Fit for Continuous Targets**  
  Predicting a house’s sale price is inherently a regression problem where the target variable is continuous. Linear regression directly models this by fitting a line (or hyperplane) to the data.

- **Interpretability of Coefficients**  
  Each learned coefficient \(w_j\) represents the change in price per unit change in feature \(x_j\), making it easy to explain which features (e.g. square footage, number of bedrooms) have the biggest impact on price.

- **Fast Training & Inference**  
  Closed-form solutions (Normal Equation) or efficient gradient-descent solvers let you train on large datasets quickly. Prediction is just a dot-product \(\mathbf{w}^\top \mathbf{x} + b\), so it’s extremely lightweight.

- **Baseline & Regularization**  
  As one of the simplest parametric models, linear regression serves as a reliable baseline. You can add Ridge (L2) or Lasso (L1) penalties to control overfitting when you incorporate many or noisy features.

- **Robust when Relationship Is Approximately Linear**  
  If your features and price are roughly linearly related (or can be made so via feature engineering), linear regression often delivers competitive performance with minimal complexity.

### How It Works

1. **Model Specification**  
   Assume the target price \(y\) is a linear combination of features plus noise:  
   \[
     \hat{y} \;=\; \mathbf{w}^\top \mathbf{x} + b
   \]

2. **Loss Function (MSE)**  
   Fit \(\mathbf{w}, b\) by minimizing the Mean Squared Error over \(n\) samples:  
   \[
     \text{MSE} = \frac{1}{n} \sum_{i=1}^n \bigl(y_i - (\mathbf{w}^\top \mathbf{x}_i + b)\bigr)^2
   \]

3. **Solution Methods**  
   - **Normal Equation (closed-form):**  
     \(\displaystyle \mathbf{w} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}\)  
   - **Gradient Descent:**  
     Iteratively update \(\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}\text{MSE}\) until convergence.

4. **Regularization (Optional)**  
   Add a penalty term to prevent overfitting:  
   - **Ridge (L2):** \(\text{MSE} + \lambda \|\mathbf{w}\|_2^2\)  
   - **Lasso (L1):** \(\text{MSE} + \lambda \|\mathbf{w}\|_1\)

5. **Prediction**  
   For a new house feature vector \(\mathbf{x}_*\), compute  
   \[
     \hat{y}_* = \mathbf{w}^\top \mathbf{x}_* + b
   \]
   to estimate its price.

By balancing simplicity, interpretability, and performance, linear regression is an ideal first choice for house-price prediction tasks.  
