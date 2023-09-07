# Basic Intro about Activation Function

- Imagine you are working as a fraud detection analyst for a bank. Your job is to identify fraudulent transactions before they are processed. You have a large dataset of historical transactions, and you want to train a neural network to learn to identify fraudulent transactions.
- Without activation functions, the neural network would simply learn a linear relationship between the input features (such as the amount of money transferred, the time of day, and the location of the transaction) and the output label (fraudulent or not fraudulent). However, most fraudulent transactions are not linearly separable from legitimate transactions. This means that a linear model would not be able to accurately identify fraudulent transactions.
- To address this problem, you can use activation functions to introduce non-linearity into the neural network. This will allow the neural network to learn more complex relationships between the input features and the output label, which will improve its ability to identify fraudulent transactions.
- **For Example**: 
    - You could use the sigmoid activation function to transform the output of each neuron into a value between 0 and 1. This would allow the neural network to learn to classify transactions as either fraudulent (output = 1) or not fraudulent (output = 0).
    - You could also use the ReLU activation function to transform the output of each neuron into a value that is greater than or equal to zero. This would allow the neural network to learn to ignore irrelevant features and focus on the most important features for fraud detection.
> Note: Activation functions in neural networks allow the network to learn non-linear relationships between the input and output data. This is important because many real-world problems involve non-linear relationships. Without activation functions, neural networks would only be able to learn linear relationships. So, By adding an activation function to each neuron, the neural network can learn to make more complex decisions. 




