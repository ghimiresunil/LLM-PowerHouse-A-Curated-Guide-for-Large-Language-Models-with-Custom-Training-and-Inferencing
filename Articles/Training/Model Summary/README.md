# Overview
- What is Deep Learning?
- Compare Neural Network with Human Brain
- What is Parameter?
- Brief Mathematical Calculation of trained Model

# Deep Learning

![deep_learning_compared_with_human_brain](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/3c453267-40b7-43a5-9727-c21a4654cb36)

- Deep Learning is a subset of machine learning.
- The algorithms of Deep Learning aim to imitate the workings of the human brain in processing data and creating patterns for use in decision making.
- Deep Learning uses the concept of artificial neural networks to carry out the process of machine learning.
- Artificial neural networks are constructed like a human brain, with neuron nodes connected together in a web-like structure.
- In our brains, a neuron consists of a body, dendrites, and an axon. The signal from one neuron travels down the axon and transfers to the dendrites of the next neuron, and this connection is called a synapse.
- Neurons are the core idea behind deep learning algorithms. We input data and pass it through hidden layers.
- The output generated by hidden layer-1 becomes input for hidden layer-2, and this continues if there are more hidden layers.
- The output of the last hidden layer is forwarded to the output layer, where the loss is calculated. The calculation of this loss is crucial during the training phase of a deep learning model. The goal is to minimize this loss by adjusting the model's parameters (weights and biases) through optimization techniques such as gradient descent.
- The most significant advantage of Deep Learning is automatic feature extraction.
- It extracts lower-level features in the initial hidden layers and higher-level features in the later layers.
- Automatically learning features at multiple levels of abstraction enables a system to learn complex function mappings from input to output directly from data, without relying solely on human-crafted features.
- The example provided above illustrates a deep learning model known as the feedforward deep network or multilayer perceptron (MLP).
- The "deep" in deep learning refers to a network with many layers.
- The learning process here is hierarchical feature learning, where each layer learns from the previous layers.

# Parameters
Parameters in general are weights that are learned during training. They are weight matrices that contribute to the model’s predictive power, changed during the back-propagation process. Who governs the change? Well, the training algorithm you choose, particularly the optimization strategy makes them change their values.

# Summary of `Model 01`

![summary_model_01](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/6f9c2dc4-1977-4943-9648-2b75e0d8db9f)

The CNN consist layer of neurons and it is optimized for two-dimensional pattern recognition. CNN has three types of layer namely convolutional layer, pooling layer and fully connected layer. Our network consists of `11 layers` excluding the input layer. The input layer takes in a RGB color image where each color channel is processed separately.

- **Convolutional Layer 1**:
  - Number of filters: 16
  - Filter size: (3, 3)
  - Input shape: (150, 150, 3)
  - Activation function: ReLU
  - Padding: 'same'
  - Parameters: (3 * 3 * 3 + 1) * 16 = 448 (3x3x3 for weights and 1 for bias)
  - Output shape: (150, 150, 16)

<br>

- **Convolutional Layer 2**:
  - Number of filters: 16
  - Filter size: (3, 3)
  - Activation function: ReLU
  - Padding: 'same'
  - Parameters: (3 * 3 * 16 + 1) * 16 = 2320 (3x3x16 for weights and 1 for bias)
  - Output shape: (150, 150, 16)

<br>

- **Convolutional Layer 3:**
  - Number of filters: 32
  - Filter size: (3, 3)
  - Activation function: ReLU
  - Padding: 'same'
  - Parameters: (3 * 3 * 16 + 1) * 32 = 4640 (3x3x16 for weights and 1 for bias)
  - Output shape: (150, 150, 32)

<br>

- **Convolutional Layer 4:**
  - Number of filters: 32
  - Filter size: (3, 3)
  - Activation function: ReLU
  - Padding: 'same'
  - Parameters: (3 * 3 * 32 + 1) * 32 = 9248 (3x3x32 for weights and 1 for bias)
  - Output shape: (150, 150, 32)


<br>

- **Convolutional Layer 5:**
  - Number of filters: 64
  - Filter size: (3, 3)
  - Activation function: ReLU
  - Padding: 'same'
  - Parameters: (3 * 3 * 32 + 1) * 64 = 18496 (3x3x32 for weights and 1 for bias)
  - Output shape: (150, 150, 64)

<br>

- **Convolutional Layer 6**:
  - Number of filters: 64
  - Filter size: (3, 3)
  - Activation function: ReLU
  - Padding: 'same'
  - Parameters: (3 * 3 * 64 + 1) * 64 = 36928 (3x3x64 for weights and 1 for bias)
  - Output shape: (150, 150, 64)

<br>

- **MaxPooling Layer**:
  - Pool size: (2, 2)
  - Output shape = (150/2, 150/2, 64) = (75, 75, 64)

<br>

- **Flatten Layer**:
  - No trainable parameters
    -  Flatten Layer doesn't introduce any new weights or biases that need to be learned during training. It doesn't perform any mathematical operations on the data that require parameter tuning. It's a non-trainable layer that only rearranges the data.
  - Output shape: 75 * 75 * 64 = 360000

<br>

- **Dense Layer 1**:
  - Number of neurons: 64
  - Activation function: ReLU
  - Parameters: (360000 + 1) * 64 = 23040 (360000 for weights and 1 for bias)
  Output shape: 64

<br>

- **Dropout Layer**:
  - Dropout rate: 0.2
  - No trainable parameters
    - The Dropout Layer does not have trainable parameters because it doesn't involve learning any new weights or biases. Its purpose is to prevent overfitting during training by randomly setting a fraction of its input values to zero. This dropout behavior is applied to the input data during each forward and backward pass through the network, but it doesn't involve learning any new parameters.
  - Output shape: 64

<br>

- **Dense Layer 2 (Output Layer)**:
  - Number of neurons: 2 (binary classification)
  - Activation function: Sigmoid
  - Parameters: (64 + 1) * 2 = 130 (64 for weights and 1 for bias)
  - Output shape: 2

<br>

- **Total trainable parameters in the model**: 448 + 2320 + 4640 + 9248 + 18496 + 36928 + 23040064 + 130 = 23112274

<br>

> Note: We train the convolutional network on the training set to find the optimal filter weights in the three convolutional layers and the weights in the two fully connected layers that minimize the error. We then evaluate the network on the validation set to get the validation error and cross-entropy loss. We repeat this process for 10 epochs and finally evaluate the network on the test set.

**CNN Architecture for first model summary**

```python
def first_model_summary():
  model = Sequential()
  model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(150,150,3)))
  model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
  
  model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
  model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
  
  model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
  model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  
  
  model.add(Flatten())
  
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(2 , activation='sigmoid'))
  
  model.compile(loss='binary_crossentropy',
                    optimizer=RMSprop(lr=0.00005),
                    metrics=['accuracy'])
  return model
```

# Summary of `Model 02`

![summary_model_02](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/a39e869c-04f9-4248-b6f0-d47745a6356b)

- **Input Layer**: The input layer has nothing to learn, at its core, what it does is just provide the input image’s shape. So no learnable parameters here. Thus a number of `parameters = 0`.
- **CONV layer**: This is where CNN learns, so certainly we’ll have weight matrices. To calculate the learnable parameters here, all we have to do is just multiply the by the shape of width m, height n, previous layer’s filters d and account for all such filters k in the current layer. Don’t forget the bias term for each of the filters. A number of parameters in a CONV layer would be : $((m * n * d)+1)* k)$, added 1 because of the bias term for each filter. The same expression can be written as follows:` ((shape of width
of the filter * shape of height of the filter * number of filters in the previous layer+1)*number offilters)`. Where the term `“filter”` refers to the number of filters in the current layer.
- **POOL layer**: This has got no learnable parameters because all it does is calculate a specific number, no backdrop learning involved! Thus a number of `parameters = 0.`
- **Fully Connected Layer (FC)**: This certainly has learnable parameters, a matter of fact, in comparison to the other layers, this category of layers has the highest number of parameters, why? because every neuron is connected to every other neuron! So, how to calculate the number of parameters here? You probably know, it is the product of the number of neurons in the current layer c and the number of neurons on the previous layer p and as always, do not forget the bias term. Thus a number of parameters here are: 
$((current\ layer\ neurons\ c\ *\ previous\ layer\ neurons\ p\ )\ +\ 1\ *\ c)$

**Now let’s follow these pointers and calculate the number of parameters, shall we?**
- The **first input layer** has no parameters.
- Parameters in the second **CONV1** `(filter shape =3*3, stride=1)` layer is:` ((shape of width of filter*shape of height filter*number of filters in the previous layer+1)*number of filters)` = (((3 * 3 * 3) + 1) * 32) = 896.
- Parameters in the fourth **CONV2** `(filter shape =3*3, stride=1)` layer is: (`(shape of width of filter * shape of height filter * number of filters in the previous layer+1) * number of filters)` = (((3 * 3 * 32) + 1)* 32) = 9248.
- The third `POOL1 layer` has no parameters.
- Parameters in the fourth **CONV3** `(filter shape =3*3, stride=1)` layer is: `((shape of width of filter * shape of height filter * number of filters in the previous layer+1) * number of filters)` = (((3 * 3 * 32) + 1) * 64) = 18496.
- Parameters in the fourth **CONV4** `(filter shape =3*3, stride=1) layer is: ((shape of width of filter * shape of height filter * number of filters in the previous layer+1) * number of filters`) = (((3 * 3 * 64) +1 ) * 64) = 36928.
- The fifth **POOL2** layer has no parameters.
- The Softmax layer has `((current layer c*previous layer p)+1*c) parameters ` = 238144 * 4 + 1 * 4 = 952580.

**CNN Architecture for second model summary**

```python
def second_model_summary():
    model = Sequential() 
    model.add(Conv2D(32, (3, 3), input_shape=(3, 256, 256), activation = 'relu')) 
    model.add(Conv2D(32, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # the CONV CONV POOL structure is popularized in during ImageNet 2014
    model.add(Dropout(0.25)) # this thing called dropout is used to prevent overfitting
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten()) 
    model.add(Dropout(0.5))
    model.add(Dense(4, activation= 'softmax'))
    optimizer = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    return model
```
