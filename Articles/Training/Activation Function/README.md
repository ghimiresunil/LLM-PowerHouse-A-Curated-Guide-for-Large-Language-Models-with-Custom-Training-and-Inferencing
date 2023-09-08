# Basic Intro about Activation Function

- Imagine you are working as a fraud detection analyst for a bank. Your job is to identify fraudulent transactions before they are processed. You have a large dataset of historical transactions, and you want to train a neural network to learn to identify fraudulent transactions.
- Without activation functions, the neural network would simply learn a linear relationship between the input features (such as the amount of money transferred, the time of day, and the location of the transaction) and the output label (fraudulent or not fraudulent). However, most fraudulent transactions are not linearly separable from legitimate transactions. This means that a linear model would not be able to accurately identify fraudulent transactions.
- To address this problem, you can use activation functions to introduce non-linearity into the neural network. This will allow the neural network to learn more complex relationships between the input features and the output label, which will improve its ability to identify fraudulent transactions.
- **For Example**: 
    - You could use the sigmoid activation function to transform the output of each neuron into a value between 0 and 1. This would allow the neural network to learn to classify transactions as either fraudulent (output = 1) or not fraudulent (output = 0).
    - You could also use the ReLU activation function to transform the output of each neuron into a value that is greater than or equal to zero. This would allow the neural network to learn to ignore irrelevant features and focus on the most important features for fraud detection.
> Note: Activation functions in neural networks allow the network to learn non-linear relationships between the input and output data. This is important because many real-world problems involve non-linear relationships. Without activation functions, neural networks would only be able to learn linear relationships. So, By adding an activation function to each neuron, the neural network can learn to make more complex decisions. 

# Sigmoid Function
<img align="right" width="400" src="https://production-media.paperswithcode.com/methods/1200px-Logistic-curve.svg_VXkoEDF.png" />

- The sigmoid function is a common choice for binary classification because it maps any input to a value between 0 and 1, which can be interpreted as a probability.
- The sigmoid function can be a good choice for some applications, but it is important to be aware of its limitations, such as gradient saturation and slow convergence.
- Pros:
    - Utilized in binary classification.
    - Offers an output that can be interpreted as a probability value since it is non-negative and in the range (0, 1).
- Cons:
    -  sharp damp gradients during backpropagation from deeper hidden layers to inputs, gradient saturation, and slow convergence.
- Usage:
    - Sigmoid functions are commonly applied in binary classification scenarios, where the output is binary, typically 0 or 1. This is because the sigmoid's output range between 0 and 1 allows for straightforward prediction: values greater than 0.5 are predicted as 1, while those less than or equal to 0.5 are predicted as 0.

- Sigmoid is defined as: $S(x)\ = \frac{1}{1 + e^{-x}}$ where, $S(x) = Sigmoid \ Function$ and $e = Euler's\ Number$

# Tanh Activation
<img align="right" width="400" src="https://api.wandb.ai/files/shweta/images/projects/57358/9914b406.png" />

- The hyperbolic tangent function, or tanh, is a popular activation function in recurrent neural networks (RNNs) and long short-term memory (LSTM) networks. It maps inputs to values between -1 and 1, which makes it suitable for modeling continuous outputs in this range.
- For Example: The tanh function is a popular activation function for RNNs and LSTMs because it can represent a wide range of values, which is necessary for modeling sequential data.
- Historically, the tanh function became preferred over the sigmoid function as it gave better performance for multi-layer neural networks. 
- But it did not solve the vanishing gradient problem that sigmoids suffered, which was tackled more effectively with the introduction of ReLU activations. [Source](https://paperswithcode.com/method/tanh-activation)
- Pros:
    - The tanh activation function is a zero-centered alternative to the sigmoid activation function, and its output range of [-1, 1] solves one of the issues with the sigmoid function.
- Cons:
    - The tanh activation function also suffers from the vanishing gradient problem, but its derivatives are steeper than the sigmoid function, which makes the gradients stronger for tanh than sigmoid.
    - As it is almost similar to sigmoid, tanh is also computationally expensive.
    - As it is almost similar to sigmoid, tanh is also computationally expensive.
- Usage:
    - Tanh is typically preferred over the sigmoid function for hidden layers because it confines its output within the range of -1 to +1, helping maintain an approximate mean of zero in the hidden layers and facilitating faster convergence during the learning process.
- Hyperbolic Tangent is defined:
$f(x) = \frac{(e^x -\ e^{-x})}{(e^x + e^{-x})}$

