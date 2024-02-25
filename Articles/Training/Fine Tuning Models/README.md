# Background 


- This readme file provides a thorough examination of fine-tuning, a widely used practice in Deep Learning.
- The file will outline the reasons for fine-tuning and the associated techniques.

# Why Do We Fine-tune Models?
- When confronted with a Deep Learning task, like training a Long Short-Term Memory (LSTM) network on a textual dataset, our initial instinct might be to begin training the network from scratch. However, in practice, LSTMs and other deep neural networks have a substantial number of parameters, often reaching millions, and training them on a small dataset (which is smaller than the number of parameters) can greatly affect the network's ability to generalize, often resulting in overfitting.
- In practical situations, it is a common practice to fine-tune pre-existing Long Short-Term Memory (LSTM) networks that were initially trained on a large dataset, such as a vast collection of textual data. This fine-tuning involves further training the LSTM using back-propagation on a smaller dataset. If our dataset is not substantially different from the original dataset (e.g., the large textual dataset), the pre-trained LSTM model will already have acquired relevant learned features suitable for our specific text classification task.

# When to Fine-tune Models?

- In general, if our textual dataset is not significantly different in context from the dataset on which the pre-trained model was originally trained, fine-tuning is recommended. A pre-trained network on a large and diverse textual dataset captures fundamental features like patterns and structures in its initial layers, which are relevant and beneficial for most text classification tasks.
- Certainly, if our textual dataset pertains to an exceedingly specialized domain, such as medical reports or Chinese medical terminology, and no pre-trained networks tailored to this domain are available, the appropriate approach would be to contemplate training the network from the scratch.
- A notable concern is that a small dataset may lead to overfitting when fine-tuning a pre-trained network, particularly if the network's final layers consist of fully connected layers, similar to the VGG network. Based on my experience, implementing common data augmentation techniques (translation, rotation, flipping, etc.) on a few thousand raw samples during fine-tuning often yields improved results.
- For extremely small datasets, containing less than a thousand samples, a more effective strategy is to use the output of the intermediate layer (bottleneck features) before the fully connected layers and train a linear classifier, such as SVM, on these features. SVM excels at creating decision boundaries on small datasets

# Fine-tuning Guidelines

Below are some general guidelines for fine-tuning implementation:
- **Truncate the last layer (softmax layer)**: 
The usual approach involves removing the last layer (softmax layer) of the pre-trained network and substituting it with a new softmax layer specific to our problem. For instance, a pre-trained network on ImageNet typically has a softmax layer with 1000 categories.
    - Suppose our task involves classifying 10 categories, the pre-trained network's new softmax layer will have 10 categories instead of the original 1000 categories. Subsequently, we perform backpropagation on the network to fine-tune the pre-trained weights while ensuring cross-validation is carried out to promote good generalization.
- **Use a smaller learning rate to train the network**: Considering that we anticipate the pre-trained weights to be substantially better compared to randomly initialized weights, we aim to avoid distorting them too rapidly or excessively. Therefore, it is a common practice to set the initial learning rate 10 times smaller than the one used for training from scratch.
- **Freeze the weights of the first few layers**: Freezing the weights of the initial layers in a pre-trained network is a common approach. These early layers capture fundamental features such as curves and edges, which remain essential for our new task. By preserving these weights, we allow the network to concentrate on learning specific features from our dataset in the subsequent layers.

# Can the user fine-tune an LLM (Language Model) to enhance performance on a predictive task such as classification or regression?

![finetune_an_LLMs](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/a7babace-c277-49a7-8915-41c238f106ed)

By now, it might be well-known that fine-tuning serves the purpose of teaching "style" rather than imparting "facts." This is particularly applicable to tasks like question answering and other generative tasks, but it may not hold true for predictive tasks.

When individuals consider fine-tuning a Language Model (LLM) to tackle classification or regression tasks, they often have the initial inclination to fine-tune both the LLM "encoder" responsible for generating embeddings and the "decoder" responsible for generating text. However, this approach can be excessive for a couple of reasons:

- Generating text output is slow
- Generating text can lead to overly verbose output and hallucinations

Instead, one can opt to remove the decoder (language model head) and replace it with a task-specific head, typically a multi-layer perceptron. This adjustment ensures that the learning objective aligns more suitably with the task at hand (for example, enabling the use of softmax cross-entropy for the loss function), and as a result, it can significantly accelerate the training process.

On the other hand, a task-specific classifier typically requires a larger number of examples for fine-tuning. Therefore, if you only possess a limited dataset of around 10 to 100 examples, it might be more advantageous to consider in-context learning or fine-tuning solely the language model head. Hence, having multiple options available for exploration based on your specific task and available data is crucial.
