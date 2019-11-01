# Simplified-gradient-descent
Optimization technique: Implementation of Gradient descent from scratch.

- Gradient descent is one of the most known optimization technique in machine learning. This technique helps in ascertaining the better estimates for the weights of the model in order to minimize the cost function values.

### Gradient descent 

- This is an optimization technique that search for better estimate for the parameters $\theta$ in order to minimize the cost function.

- With gradient descent there two major parameters to tune which can be hyperparameter tuning. Those parameters are: 
  - Number of iteration 
  - learning rate $\alpha$

$$h_{\theta}(x) = \sum_{i=0}^{m} \theta_{i}x_{i} = \theta^{T}X$$

- We set the bias also know intercept to $x_{0}=1$.  <br>

In linear regression we normally define a function to measure the difference between $h_{\theta}(x^{i})$ and $y^{i}$ know as linear cost function. Here is the cost function defined below;

$$ Cost(\theta) = \dfrac{1}{2m}\sum_{i=0}^{m}(y^{i} - h_{\theta}(x^{i})^{2}$$

The function can be derive using maximum likelihood estimation approach.

In order to ascertain $\theta$ value which minimize the cost function, we have to define the learning rule for $\theta$ which is known as Widrow-Hoff learning rule. Define below:

\begin{eqnarray}
\theta_{k} := \theta_{k-1} + \dfrac{\alpha}{m} (y^{(i)}-h_{\theta}(x^{i}))x_{k}^{i} 
\end{eqnarray}

where the following parameters is defined as follow:<br>
$\alpha$ is the learning rate of the algorithm. <br>
k represents $ \theta_{new}$ is the new weight update for each feature.<br>
k-1 represents $ \theta_{old}$ is the previous weight for each feature.<br>

The above algorithm is for single update rule in a scenario but in a given vector of $\theta$ we define it as :.
$$\theta_{k} := \theta_{k-1} + \dfrac{\alpha}{m}\sum_{i=1}^{m}(y^{(i)}-h_{\theta}(x^{i}))x_{k}^{i} $$

To this data, we had to select five features to demonstrate how the gradient work. In most cases we have scale the dataset but in our case we did not scale features since we want to illustrate how to implement the gradient descent from the scratch. We observed that the appropriate learning is $\alpha = e^{-5}$  but for the number of iteration it depends on the number of times you want to train the parameters in order to have better estimate of the weights which can minimize the cost estimate of the model. 

- We initialize the weights using the normal distribution rather than setting the parameters to zeros. 

Here is the source: [Reference](http://cs229.stanford.edu/notes2019fall/cs229-notes1.pdf)
