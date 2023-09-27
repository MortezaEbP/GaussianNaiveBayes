# GaussianNaiveBayes
# Gaussian Naive Bayes Algorithm
Gaussian Naive Bayes is a probabilistic classification algorithm that makes use of Bayes' theorem and assumes that features are normally distributed within each class. This description provides an overview of the algorithm's mathematical formulation and highlights its vectorized computations.
## Mathematical Formulation

Gaussian Naive Bayes predicts the class label y for an input feature vector x by computing the posterior probability using Bayes' theorem:


$${\Huge\displaystyle \Huge \ P(y | x) = \frac{P(x | y) \cdot P(y)}{P(x)}}$$

### Where:

${\mathbf{P(y | x)}}$ is the posterior probability of class ${\mathbf{y}}$ given feature ${\mathbf{X}}$.

${\mathbf{P(x | y)}}$ is the likelihood of observing feature ${\mathbf{X}}$ given class ${\mathbf{y}}$.

${\mathbf{P(y)}}$ is the prior probability of class ${\mathbf{y}}$.

${\mathbf{X}}$ is the probability of observing feature ${\mathbf{X}}$.


For Gaussian Naive Bayes, the likelihood ${\mathbf{P(x | y)}}$ is assumed to follow a Gaussian distribution:


$${\Huge\displaystyle \Huge \ P(x | y) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma_{y,i}^2}} \exp\left(-\frac{(x_i - \mu_{y,i})^2}{2\sigma_{y,i}^2}\right)\}$$

${\mathbf{n}}$ is the number of features.

${\mathbf{x_i}}$ is the $\(i\)$-th feature of ${\mathbf{x}}$.

$\mu_{y,i}\$ is the mean of the $\(i\)$-th feature for class y.

$\sigma_{y,i}^2\$ is the variance of the $\(i\)$-th feature for class y.



## Vectorized Computations
To enhance computational efficiency, Gaussian Naive Bayes can be implemented using vectorized operations. These operations leverage NumPy arrays and perform calculations across entire arrays, enabling faster processing.

# Conclusion
Gaussian Naive Bayes is a simple yet effective algorithm for classification tasks, especially when the independence assumption holds between features. It leverages the probabilistic framework of Bayes' theorem and assumes Gaussian distribution of features within classes. By utilizing vectorized computations, Gaussian Naive Bayes efficiently calculates probabilities and makes it suitable for large datasets and high-dimensional feature spaces.

For a practical implementation of Gaussian Naive Bayes in Python, along with code and usage examples, please refer to the provided code repository. The repository includes detailed explanations and sample code to assist you in understanding and applying the algorithm effectively.
