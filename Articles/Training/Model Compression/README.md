# Enhancing Model Compression: Inference and Training Optimization Strategies

- Are you familiar with the implementation of model compression techniques for both training and inference in various machine learning and deep learning models? If not then you will acquire knowledge about various model compression and optimization concepts applicable during both model training and inference. These concepts encompass quantization/binarization, pruning, knowledge distillation, mixed precision training, and quantization-aware training.
- The exponential increase in the number of parameters in machine learning and deep learning models have become larger and larger, the computational requirements for training and inferencing them have also increased exponentially. This can make them impractical for use in edge environments or for serving to customers at scale.
- This article highlights state-of-the-art techniques developed by researchers and practitioners to enhance the efficiency of machine learning and deep learning models, enabling them to run more swiftly and with reduced memory consumption on edge devices 

# Quantization
## Background: Precision
- Before we talk about quantization, let’s learn about precision [Source](https://blogs.nvidia.com/blog/2019/11/15/whats-the-difference-between-single-double-multi-and-mixed-precision-computing/).
    - Precision in computer science is a measure of the accuracy of a numerical value. It is typically measured in bits, with more bits representing a higher precision. For example, a single-precision floating-point number has 32 bits, while a double-precision floating-point number has 64 bits. This means that a double-precision number can represent a wider range of values and with more accuracy than a single-precision number.
    - The precision of a calculation can be affected by the number of digits used to represent the input values. For example, if you calculate pi using only two digits to the right of the decimal point (3.14), you will get a less accurate result than if you use ten digits (3.1415926535). This is because the computer has less information to work with when there are fewer digits.
    - Here is another example to illustrate the concept of precision in computer science. Let's say you want to calculate the area of a circle with a radius of 10. If you use a single-precision floating-point number to represent the radius, the area will be calculated as 314.159265. However, if you use a double-precision floating-point number, the area will be calculated as 314.1592653589793. The difference between these two values is due to the fact that the double-precision number has more precision and can therefore represent the radius more accurately.
- [The Institute of Electrical and Electronics Engineers (IEEE)](https://standards.ieee.org/ieee/754/6210/) developed the IEEE Standard for Floating-Point Arithmetic (IEEE 754) to ensure that there are no huge discrepancies in calculations when representing large numbers in computer binary.
    - The base - Indicates whether the number is positive (0) or negative (1).
    - The biased exponent - The exponent is used to represent both positive and negative exponents. Thus, a bias must be added to the actual exponent to get the stored exponent. 
    - The mantissa - Also known as the significand, the mantissa represents the precision bits of the number.

> Note: IEEE 754 is the most common way to represent floating-point numbers because it is the most efficient way to represent numerical values. It has two formats: single-precision and double-precision.

- Difference Between Single and Double Precision

| Aspect | Single-Precision (FP32) | Double Precision (FP64) |
| ------- | ---------------- | ---------------- |
| Number of bits | 32  | 64 | 
| Sign Bit | 1 bit | 1 bit | 
| Exponent Bits | 8 bits | 11 bits |
| Mantissa (Significand) Bits | 23 bits | 52 bits | 
| Exponent Bias | 127 | 1023 | 
| Precision (Significant Digits) | Approximately 7 decimal digits	| 	15 decimal digits | 
| Approximation Errors	| More noticeable due to lower precision | Less noticeable due to higher precision | 
| Memory Usage | Less memory usage per number	| More memory usage per number |
| Performance | Faster calculations	| Slower calculations|
| Example <br> Take Euler’s number (e), for example. Here are the first 50 decimal digits of e: <br> 2.7182818284590452353602874713526624977572470936999| Euler’s number in binary, converted to single precision <br> `0` `10000000` `01011011111100001010100` <br> Sign Bit: `0` <br> Exponent Bits: `10000000` <br> Mantissa: `01011011111100001010100`|  Euler’s number in binary, converted to double precision <br> `0` `10000000000010110111111` `0000101010001011000101000101011101101001` <br> Sign bit: `0` <br> Exponent Bits: `10000000000010110111111` <br> Mantissa Bits: `0000101010001011000101000101011101101001`| 
| Use Cases | Graphics, real-time simulations, gaming | Scientific computations, engineering | 

> Note: Half precision (FP16) takes an even smaller, with just five for bits for the exponent and 10 for the significand. <br> Euler’s number in binary, converted to half precision: `0` `10001` `1011001100` <br> Sign Bit: `0` <br> Exponent Bits: `10001` <br> Mantissa: `1011001100` 

- Difference Between Multi-Precision and Mixed-Precision Computing

| Aspect | Multi-Precision Computing | Mixed-Precision Computing |
| -------- | -------------------- | ------------------- |
| Precision Levels | Utilizes multiple precision formats, such as single (32-bit) and double (64-bit) precision. | Primarily starts with low precision, typically half (16-bit), and transitions to higher precision as needed. Commonly a combination of half, single, and double precision. | 
| Use Cases	| Widely used in various numerical applications requiring high accuracy and precision.	| Primarily employed in machine learning and deep learning, where rapid calculations with a balance between accuracy and performance are essential. | 
| Accuracy vs Performance | Emphasizes higher accuracy and precision, suitable for scientific simulations, cryptography, and other domains where accuracy is critical. | Balances between accuracy and performance by using lower precision for initial computations and gradually increasing precision as needed. Sacrifices some accuracy for increased performance. | 
| Computational Cost | More computationally intensive due to higher precision calculations, leading to potentially increased power consumption, runtime, and memory usage. | Offers a trade-off between computational cost and accuracy. Low precision computations reduce computational load and memory usage, contributing to better efficiency.| 
| Memory Requirements | Generally requires more memory due to the larger storage needs of higher precision data types. | Requires less memory initially due to lower precision storage, with potential memory growth as precision is increased during calculations. | 
| Applications | Common in fields like scientific computing, numerical simulations, and financial modeling. | Predominantly used in machine learning training and inference, neural network computations, and other tasks where optimization of performance is crucial. | 
- Benefits of mixed-precision computing
    - Faster computation: Uses lower precision for initial calculations, which are faster than higher precision calculations.
    - Reduced memory usage: Lower precision data types require less memory than higher precision ones.
    - Increased throughput: Faster computations and reduced memory usage lead to increased throughput.
    - Energy efficiency: Lower precision calculations consume less energy than higher precision ones.
    - Scalability: Allows for scaling up the training of large neural networks.
    - Fine-tuned precision: Enables finer control over precision for specific operations.
    - Compatibility: Supported by many modern GPUs and specialized hardware accelerators.
    - Similar accuracy: Can achieve accuracy similar to that of higher precision computations.
    - Reduced overhead: Transitions between different precision levels with minimal overhead.
    - Customizable trade-offs: Practitioners can adjust the balance between precision and performance.

## Definition
- Quantization involves reducing the bit precision of model parameters (**weights** and often **activations**) from high precision (32 or 64 bits) to lower bit representations (like 16 or 8 bits), resulting in a speedup of around 2-4 times, particularly noticeable in networks with convolutional operations. Furthermore, quantization achieves around 4x model compression
- Quantization can cause a model to lose accuracy. "Quantization-aware training" fine-tunes the model after quantization to regain accuracy. "Post-training quantization" skips fine-tuning and applies heuristics to the quantized weights to try to preserve accuracy. Both methods aim to shrink model size with minimal impact on accuracy. Fine-tuning counteracts the performance drops caused by reduced precision, making quantization a valuable model optimization.
- Deep learning models can be made faster and more memory efficient by using fewer bits to represent their weights. This is because there are fewer bits to compute when using fewer bits to represent the weights. For example, using `torch.qint8`, which is an 8-bit integer type, instead of `torch.float32`, which is a 32-bit floating-point type, can make a deep learning model 4 times faster and 4 times more memory efficient.

> Potential downsides involve situations where certain operations you require—like particular convolutional processes or even simple tasks like transposition—might not be available due to the degree of quantization you're aiming for. Additionally, accuracy may drop too much when quantization is used. This can make quantized models less useful for certain applications.
- From TensorFlow Document [Source](https://www.tensorflow.org/model_optimization/guide/quantization/post_training#:~:text=training%20float16%20quantization-,Quantizing%20weights,bit%20integer%20for%20CPU%20execution.&text=At%20inference%2C%20the%20most%20critically,bits%20instead%20of%20floating%20point.)
> We generally recommend 16-bit floats for GPU acceleration and 8-bit integer for CPU execution.

## Quantization with PyTorch
- PyTorch supports quantized tensors, which are tensors that store data in a more compact format, such as 8 or 16 bits. This can be useful, but it is important to understand how this works in order to ensure that the model still performs well. If your neural network has a sigmoid activation function, then you can use a quantization method that is specifically designed for the output range of 0 to 1. To quantize a neural network, you need to collect data about how the network performs on a set of representative inputs. This data will be used to choose the best quantization method for the network. Quantization is typically performed by rounding $x * scalar$, where $scalar$ is a learned parameter, similar to $BatchNorm$.

> PyTorch supports quantized operations through external libraries, such as FBGEMM and QNNPACK. These libraries provide efficient implementations of quantized operations that can be loaded as needed, similar to BLAS and MKL. 

- Quantization can introduce numerical instability, especially if the input data has a large exponent. In these cases, it is often better to accumulate in higher precision data types to improve stability. Choosing the right precision for each operation in a PyTorch neural network can be difficult. The `torch.cuda.amp` package provides automatic mixed precision (AMP) functionality to help you cast different parts of your network to half precision `(torch.float16)` where it is possible. This can improve performance without sacrificing accuracy. If you want to manually choose the precision for each operation in your PyTorch neural network, you can find some helpful guidance on the [PyTorch: Automated Mixed Precision page](https://pytorch.org/docs/stable/amp.html).

> You can try running your existing model `torch.flot32` in mixed precision using `torch.cuda.amp` to see if it still maintains its accuracy Half-precision support is not widely available in consumer GPUs, but it is supported by the V100, P100, and A100 GPUs,

- There are three levels of manual quantization available in PyTorch, called eager mode quantization. These levels offer different levels of control and flexibility, depending on your needs. If you want to deploy your model to a non-CUDA environment or have more control over the quantization process, you can use eager mode quantization.

| Quantization Method |	When to Use	 | Pros | Cons| 
| -------------------- | -------------|-------------|-------------|
| Dynamic quantization | When you need to quantize a model quickly and easily, without sacrificing too much accuracy. | Simple to implement, requires no changes to the model training process. | Can lead to loss of accuracy for models with large activations. | 
| Static quantization | When you need to achieve the highest possible accuracy after quantization. | Can achieve higher accuracy than dynamic quantization, especially for models with large activations. | Requires a calibration step after training, which can be time-consuming. | 
| Static quantization-aware training | When you want to combine the benefits of dynamic quantization and static quantization. | Can achieve high accuracy with faster training times than static quantization. | Requires more complex training process than dynamic quantization. | 

- For a more comprehensive overview of the tradeoffs between quantization types, please see the PyTorch blog post "[Introduction to Quantization](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)".

> The coverage of layers and operators (such as Linear, Conv, RNN, LSTM, GRU, and Attention) for dynamic and static quantization differs, as shown in the table below. FX quantization also supports the corresponding functionals.

### Dynamic/Runtime Quantization

- Dynamic quantization is a simple and effective way to quantize models in PyTorch. Dynamic quantization in PyTorch involves converting the weights to int8, as with other quantization methods, but it also converts the activations to `int8` on the fly, just before the computation is performed. The computations will be performed using efficient `int8` matrix multiplication and convolution implementations, which will result in faster compute. However, the activations will be read and written to memory in floating point format.
- The network's weights are stored in a specified quantization format. At runtime, the activations are dynamically converted to the same quantization format, combined with the quantized weights, and then written to memory in full precision. The output of each layer in a quantized neural network is quantized and then combined with the quantized weights of the next layer. This process is repeated for each layer in the network, until the final output is produced. I am confused why this happens, my understanding is that `scalars` could be dynamically determined from the data, which would mean that this is a data-free method.
- To quantize a model in PyTorch, we can use the `torch.quantization.quantize_dynamic` API. This API takes in a model and a few other arguments, and returns a quantized model. For an example of how to use this API, see this [end-to-end](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html) tutorial on quantizing a BERT model.

```python
# quantize the LSTM and Linear parts of our network
# and use the torch.qint8 type to quantize

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
```
- There are many ways to improve your model by adjusting its hyperparameters. More details can be found in this [blog post](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html).
- For more information on the function, please see the documentation [here](https://pytorch.org/docs/stable/quantization.html#torch.quantization.quantize_dynamic). For an end-to-end example, please see tutorials [here](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html) and [here](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html).

### Post-Training Static Quantization
- Converting activations to full precision and back at runtime is computationally expensive. However, we can avoid this cost if we know the distribution of activations, which can be determined by recording real data flowing through the network. 
- Neural networks can be made to perform faster (in terms of latency) by converting them to use bot integer arithmetic and `int8` memory accesses. Static quantization involves feeding batches of data through the network to compute the distribution of activations. This is done by inserting observer modules at different points in the network to record the distributions.The information is used to determine how to quantize the activations at inference time. We can use a simple method that divides the activations into 256 levels, or we can use a more sophisticated method.
- By quantizing the values, we can avoid the overhead of converting them to floats and then back to ints, which can significantly improve performance.
- Users can use the following features to optimize their static quantization models for better performance.
    - **Observers**: By customizing observer modules, you can control how statistics are collected prior to quantization, which can lead to more accurate and efficient quantization of your data.
        - Observers are inserted using `torch.quantization.prepare`.
    - **Operator fusion**: PyTorch can inspect your model and implement additional optimizations, such as quantized operator fusion, when you have access to data flowing through your network. This can save on memory access and improve the operation's numerical accuracy by fusing multiple operations into a single operation.
        - To fuse modules, use `torch.quantization.fuse_modules`.
    - **Per-channel quantization**: By quantizing the weights of a convolution or linear layer independently for each output channel, we can achieve higher accuracy with almost the same speed.
        - Quantization itself is done using `torch.quantization.convert`.
- Below is an example of how to set up observers, run them with data, and export a new statically quantized model.

```python
# this is a default quantization config for mobile-based inference (ARM)
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
# or set quantization config for server (x86)
# model.qconfig = torch.quantization.get_default_config('fbgemm')

# this chain (conv + batchnorm + relu) is one of a few sequences 
# that are supported by the model fuser 
model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

# insert observers
model_with_observers = torch.quantization.prepare(model_fused)

# calibrate the model and collect statistics
model_with_observers(example_batch)

# convert to quantized version
quantized_model = torch.quantization.convert(model_with_observers)

```
### Static Quantization-aware Training (QAT)
- Quantization-aware training (QAT) is the third method for quantization, and it typically achieves the highest accuracy. In QAT, weights and activations are "fake quantized" i.e. rounded to `int8` values during both the forward and backward passes of training, while computations are still done with floating point numbers. This allows the model to be trained with the knowledge/aware that it will be quantized, resulting in higher accuracy than other methods.
- QAT tells the model about its limitations in advance, and the model learns to adapt to these limitations by rounding its activations to the chosen quantization during the forward and backward passes. This helps the model to learn to be more robust to quantization, resulting in higher accuracy after quantization.
> During QAT, the backpropagation (gradient descent of the weights) is performed in full precision. This is important because it ensures that the model is able to learn the correct weights, even though the activations are being rounded to a lower precision.
- To quantize a model, `torch.quantization.prepare_qat` first inserts fake quantization modules to the model. Then, `torch.quantization.convert` quantizes the model once training is complete, mimicking the static quantization API.
- In the [end-to-end example](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html), we load a pre-trained model into a variable called qat_model. Then, we perform quantization-aware training on the model using the below code:
```python
# specify quantization config for QAT
qat_model.qconfig=torch.quantization.get_default_qat_qconfig('fbgemm')

# prepare QAT
torch.quantization.prepare_qat(qat_model, inplace=True)

# convert to quantized version, removing dropout, to check for accuracy on each
epochquantized_model=torch.quantization.convert(qat_model.eval(), inplace=False)
```
> It is recommended to read the helpful tips under the “[Model Preparation for Quantization](https://pytorch.org/docs/stable/quantization.html)” section of the PyTorch documentation before using PyTorch quantization.

## Device and Operator Support
- Quantization support is limited to a subset of operators, depending on the quantization method. For a list of supported operators, please see the [documentation](https://pytorch.org/docs/stable/quantization.html).
- The set of operators and quantization methods available for quantized models depends on the backend used to run them. Currently, quantized operators are only supported for CPU inference on the x86 and ARM backends. The quantization configuration and quantized kernels are also backend dependent. To specify the backend, you can use:

```python
import torchbackend='fbgemm'

# 'fbgemm' for server, 'qnnpack' for mobile
my_model.qconfig = torch.quantization.get_default_qconfig(backend)

# prepare and convert model
# Set the backend on which the quantized kernels need to be run
torch.backends.quantized.engine=backend
```
- Quantization-aware training is a process that trains CNN models in full floating point, and can be run on either GPU or CPU. It is typically used when post-training quantization does not yield sufficient accuracy, such as with models that are highly optimized to achieve small size.

## Integration in Torchvision
- PyTorch has made it easier to quantize popular models in [torchvision](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) by providing quantized versions of the models, a quantization API, and a quantization tutorial. This makes it possible to deploy quantized models to production with less effort.
    - Quantized versions of the models, which are pre-trained and ready to use.
    - Quantization-ready model definitions are provided so that you can quantize a model after training (post-training quantization) or quantize a model during training (quantization aware training).
    - Quantization aware training is a technique that can be used to improve the accuracy of any model, but we found it to be especially beneficial for Mobilenet
    - You can find a [tutorial](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html) on that demonstrates how to do transfer learning with quantization using a torchvision model.

## Choosing an Approach
- The decision of which scheme to use is dependent by a variety of factors.
    - **Model/Target requirements**: Some models are more sensitive to quantization than others, and may require quantization-aware training to maintain accuracy.
    - **Operator/Backend support**: There are backends that only work with fully quantized operators.
- The number of quantized operators available in PyTorch is currently limited, which may impact the choices you can make from the table below. This table, from PyTorch: Introduction to Quantization on PyTorch, provides some guidance.

## Performance Reults
- Quantization can reduce the model size by 4x and speed up inference by 2x to 3x, depending on the hardware platform and the model being benchmarked. The table below from the [PyTorch documentation on quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) provides some sample results of the technique.

## Accuracy Results
- The tables in [PyTorch's Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) document compare the accuracy of quantized models to floating-point models on the ImageNet, as well as we compared the F1 score of BERT on the GLUE benchmark for MRPC.

### Computer Vision Model Accuracy
### Speech and NLP Model Accuracy

## Quantization in Other Frameworks: TensorFlow and CoreML
- PyTorch quantization may not work in all production environments, such as when converting a model to Apple's CoreML format, which requires 16-bit quantization. When deploying a model to an edge device, it is important to check that the device supports quantization. On Apple devices, the hardware already computes everything in `fp16`, so quantization is only useful for reducing the memory footprint of the model.
- TensorFlow uses a similar set of steps as above, but the examples are focused on TFLite. 
- The [post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) page explains static and dynamic quantization, and the QAT page provides more information about quantization aware training. The tradeoffs between PyTorch and TensorFlow for quantization are similar, but there are some features that are not compatible between the two frameworks.

## How Far Can We Go?
- Researchers have been working on binary neural networks [for years](https://arxiv.org/abs/1909.13863), as they offer the potential for extreme speedups with only a small loss in accuracy. These networks use only 1 bit of precision for their weights and activations, which can be much faster to compute than traditional neural networks. Binary neural networks are still mostly research projects, but [XNOR-Net++](https://arxiv.org/abs/1909.13863) is a notable exception as it has been implemented in PyTorch. This makes it a more usable idea for practical applications.

## Use-case
- Quantization is a technique that can be used to increase the speed of model inference by reducing the precision of the model's parameters. In contrast, please reed Mixed Precision Training, Automatic Mixed Precision (AMP) is a technique that can be used to reduce the training time of a model by using lower precision numbers during training.

# Knowledge Distillation
- Knowledge distillation is a technique for transferring knowledge from a large model (teacher) to a smaller model (student), resulting in smaller and more efficient models. [Hinton et al., 2015](https://arxiv.org/abs/1503.02531)
- "Knowledge distillation is a process of transferring knowledge from a large model (teacher) to a smaller model (student). The student model can learn to produce similar output responses (response-based distillation), reproduce similar intermediate layers (feature-based distillation), or reproduce the interaction between layers (relation-based distillation)." [aiedge.io](https://newsletter.theaiedge.io/)
- The image below, which is sourced from [AiEdge.io](https://newsletter.theaiedge.io/), does an excellent job of visualizing the concept of knowledge distillation.
- Knowledge distillation is a technique that allows us to deploy large deep learning models in production by training a smaller model (student) to mimic the performance of a larger model (teacher).
> The key idea of knowledge distillation is to train the student model with the soft target of the teacher model's output probability distribution, instead of the same labeled data as the teacher.
- During a standard training process, the teacher model learns to discriminate between many classes by maximizing the probability of the correct label. This side effect, where the model assigns smaller probabilities to other classes, can give us valuable insights into how the model generalizes. For example, an image of a cat is more likely to be mistaken for a tiger than a chair, even though the probability of both mistakes is low. We can use this knowledge to train a student model that is more accurate and robust.
- The student model is typically a smaller version of the teacher model, with fewer parameters. However, it is recommended to use the same network structure as the teacher model, as this can help the student model to learn more effectively. For example, if we want to use BERT as a teacher model, we can use DistillBERT, which is a 40% smaller version of BERT with the same network structure.
- The student model is trained to minimize a loss function that is a combination of the teacher's original training loss and a distillation loss. The distillation loss is calculated by taking the teacher's softmax output for the correct class, averaging it with the softmax output of the student model, and then scaling the result with a temperature parameter. The temperature parameter controls how soft the averaging is, with a higher temperature resulting in a softer averaging.
-  The temperature parameter effectively smooths out the probability distribution, reducing the higher probabilities and increasing the smaller ones. This results in a softer distribution that contains more knowledge about the uncertainty of the prediction.
- Knowledge distillation can significantly reduce the latency of a machine learning model, with only a small decrease in accuracy.
- In practice, for a classification task, we can train a smaller student model $f_{\theta}$, where $\theta$ is the set of parameters, by using a large model or an ensemble of models (possibly even the same model trained with different initializations), which we call $F$. We train the student model with the following loss function:
$$\mathcal{L} = \sum\nolimits_{i=1}^n KL(F(x_i),f_{\theta}(x_i))$$

where, $F(x_i)$ = probability distribution over the labels created by passing example $x_i$
 through the network
 - You can optionally add the regular cross-entropy loss to the loss function by passing in the one-hot ground truth distribution to the student model as well. 
 $$\mathcal{L} =  \sum\nolimits_{i=1}^n(KL(F(x_i), f_{\theta}) - \beta.\sum\nolimits_{=1}^K y_i[k] logf_{\theta}(x_i)[k])$$
> Note: The second term in the loss function is the $KL$ Kullback-Leibler divergence from the one-hot distribution of the labels (the "true" distribution) to the student model's distribution, since the one-hot distribution is a special case of the softmax distribution.
- There is no consensus on why knowledge distillation works, but the most compelling explanation is that it is a form of data augmentation. This paper, [Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning](https://arxiv.org/abs/2012.09816), provides a good explanation of why this is the case. The paper is focused on the idea of multiple views, and it provides some thought experiments that may help to explain what is happening at a deeper level.

## Distillation Thought Experiment
- A teacher model trained to classify images may have filters that are sensitive to pointy ears. These filters may fire even when the model is presented with an image of a Batman mask, which is not a cat. This suggests that the model thinks the Batman mask looks 10% like a cat.
- A student model trained to match the probability distribution of the teacher will learn that the Batman mask has a 10% probability of being a cat. This information can help the student model recognize cat-like images, even if they are not labeled as cats. This is because the student model is able to learn from the teacher model's mistakes and identify images that are similar to cats, even if they do not have the exact same features. This logic also explains why the student model can sometimes outperform the teacher model, as the student model is able to learn from the teacher model's mistakes and make better predictions.

## Ensembling Thought Experiment
- Ensembles of models (even with the same architecture) can work well because they can learn different features from the same data. For example, in a dataset of cat images, one model might learn to identify cats with pointed ears, another model might learn to identify cats with whiskers, and a third model might learn to identify cats with both features. By combining the predictions of multiple models, ensembles can reduce the risk of overfitting and improve the overall accuracy of image classification.
- Neural networks can learn to recognize features in data, but they may not learn all of the features that are important. For example, a neural network might learn to recognize feature A by seeing image 1. However, if the network only sees image 1 and image 3, it may not learn feature B, even though feature B is also present in image 3. This is because the network will not receive any gradient signal to learn feature B. A good neural network would learn both feature A and feature B, but this may not always happen.
-  A neural network learns to classify data points more accurately, it may become less sensitive to small changes in the data. This is because the network is already able to classify the data points with high confidence, so it does not need to rely on every single data point to make a decision. As a result, the signal from some data points may decrease as the network becomes more accurate.

## Distillation in Practice
- Knowledge distillation is a rapidly growing research field with applications in defending against adversarial attacks, transferring knowledge between models, and protecting privacy.
- In knowledge distillation, the student model learns from the teacher model by mimicking its predictions. Response-based distillation focuses on the final output layer of the teacher model, while feature-based and relation-based distillation focus on other parts of the teacher model.
- The different types of distillation, such as offline distillation (where the student model is trained after the teacher model), online distillation (where the student and teacher models are trained together), and self-distillation (where the teacher model has the same architecture as the student model), can make it difficult to track distillation in practice. A set of ad hoc model-specific techniques may be the best general recommendation.
- In Fact,
    - [Cho & Hariharan (2019)](https://arxiv.org/abs/1910.01348) found that knowledge distillation can be harmful when the student model is too small. They also found that knowledge distillation papers rarely use ImageNet and so often don't work well on difficult problems.
    - [Mirzadeh et al. (2019)](https://arxiv.org/abs/1902.03393) found that better teacher models don't always mean better distillation, and that the farther the student and teacher model's capacities are, the less effective distillation is.
    - A recent investigation by [Tang et al. (2021)](https://arxiv.org/pdf/2002.03532.pdf) supports these findings.
> Note: Knowledge distillation can be harmful when the student model is too small, and it is less effective when the student and teacher models have different capacities.
- In Summary, knowledge distillation is a powerful technique for improving the performance of small models, but it is more difficult to implement than quantization and pruning. However, it can be worth the effort if you need to achieve a high level of accuracy with a small model.

### Distillation As Semi-supervised Learning
- A teacher model can be used to transfer knowledge to a student model. The teacher model is first trained on a large set of labeled data. Then, it is used to generate soft labels for a smaller set of unlabeled data. These soft labels can then be used to train the student model. This approach allows the student model to learn from the knowledge of the teacher model, even though it is not trained on as much data.
- [Parthasarathi and Strom (2019)](https://arxiv.org/pdf/1904.01624.pdf) used a two-step approach to train an acoustic model for speech recognition. First, they trained a powerful teacher model on a small set of annotated data. This teacher model was then used to label a much larger set of unannotated data. Finally, they trained a leaner, more efficient student model on the combined dataset of annotated and unlabeled data.

# Pruning
- Pruning is a way to remove unnecessary weights or neurons from a neural network, making it smaller and faster without sacrificing accuracy. We can often prune up to [90% of the parameters](https://arxiv.org/abs/1506.02626) in a large deep neural network without any noticeable loss in performance.
- Model pruning is a way to remove unnecessary weights from a neural network by understanding which weights are important to the model's performance. This can be done by analyzing the impact of each weight on the loss function, using regularization methods (L1 and L2), or removing entire neurons or layers. The goal of pruning is to create a smaller, more efficient model without sacrificing accuracy.
- The [lottery ticket hypothesis](https://arxiv.org/abs/1803.03635) is a theory that explains why model pruning works. It states that every neural network contains a subnetwork that is capable of achieving the same accuracy as the original network, even if it is much smaller. This subnetwork is called a "winning ticket."
    - For example: Let's say we have a tabular dataset with 100 features and 10 classes. We want to train a neural network to predict the class of each data point.

        We could train a large neural network with 1000 neurons in the hidden layer. This network would have 100,000 parameters. However, according to the lottery ticket hypothesis, this network contains a subnetwork with only a few thousand parameters that can be trained to achieve the same accuracy as the original network.

        We could find this subnetwork by using a technique called pruning. Pruning is a process of removing the less important parameters from a neural network. This can be done by iteratively removing the parameters with the smallest weights or by removing the parameters that have the least impact on the network's performance.

        In this example, we could use pruning to remove 95% of the parameters from the original network. This would leave us with a smaller network with only 5000 parameters. However, this smaller network would still be able to achieve the same accuracy as the original network.

    - Randomly initialized neural networks contain subnetworks that can be trained to achieve similar accuracy to the original network, even if they are much smaller. This is known as the lottery ticket hypothesis.

> Large neural networks contain subnetworks that can be trained to achieve the same accuracy as the original network, even if they are much smaller.

## Structured vs. Unstructured Pruning
- Structured pruning removes neurons or chooses a subnetwork, while unstructured pruning sparsifies model weights using methods such as TensorFlow's `tensorflow_model_optimization` and PyTorch's `torch.nn.utils.prune`. This can save disk space using compression algorithms such as run-length encoding or [byte-pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding) and it may also speed up inference when sparse model support is fully implemented in various frameworks because multiplying a sparse vector and a sparse matrix is faster than multiplying a dense vector and a dense matrix.
- Structured pruning, a dynamic research field lacking a clear API, involves selecting a metric to assess the significance of each neuron. Subsequently, neurons with lower information content can be pruned, with potentially useful metrics encompassing the [Shapley value](https://christophm.github.io/interpretable-ml-book/shapley.html), a Taylor approximation measuring a neuron's impact on loss sensitivity, or even random selection. Notably, the [TorchPruner](https://github.com/marcoancona/TorchPruner) library automatically incorporates some of these metrics for `nn.Linear` and convolution modules, while the [Torch-Pruning](https://github.com/vainf/torch-pruning) library offers support for additional operations. Among the notable earlier contributions, one involves filter pruning in convnets using the L1 norm of filter weights.
- Unstructured pruning is a technique for reducing the size of a neural network by zeroing out weights with small magnitudes. It can be done during or after training, and the target sparsity can be adjusted to achieve the desired balance between model size and accuracy. However, [there is some confusion](https://arxiv.org/abs/2003.03033) in this area, so it is important to consult the documentation for [TensorFlow](https://www.tensorflow.org/model_optimization/guide/pruning/) and [PyTorch](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) before using unstructured pruning.

## Fine Tuning | [What is Fine Tuning](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/tree/main/Articles/Training/Fine%20Tuning%20Models)
- After pruning a neural network, it is [standard practice](https://arxiv.org/pdf/2003.02389.pdf) to retrain the network. The best method is to reset the learning rate to its original value and start training from scratch. Optionally, you can also reset the weights of the unpruned parts of the network to their values earlier in training. This is essentially training the lottery ticket subnetwork that we have identified.
    - For example: let's say we have a neural network with 1000 weights. We use pruning to remove 90% of the weights, leaving us with 100 weights. We then retrain the network with a reset learning rate and the weights from the earlier training. This helps the lottery ticket subnetwork to learn how to perform the task at hand more effectively.

>  If you are interested in network pruning, you can start by using [TorchPruner](https://github.com/marcoancona/TorchPruner) or [Torch-Pruning](https://github.com/vainf/torch-pruning) to prune the network. Then, you can fine-tune the resulting network with learning rate rewinding. However, it is not always clear how to trim the rest of the network around the pruned part, especially for architectures with skip connections like ResNets.

# DeepSpeed and ZeRO-Offload
- [DeepSpeed](https://www.deepspeed.ai/) is a library that optimizes the training of large and extremely large models on GPUs. It does this by using smart parallelism and better caching, which can lead to significant speedups and memory savings. DeepSpeed is an extension to PyTorch, so it is easy to use with existing PyTorch code.

# Mixed Precision Training

## Overview
- Mixed precision training is a technique that uses half-precision floating point numbers (`float16`) to speed up neural network training while maintaining accuracy. By using `float16` for most operations, the training time is reduced significantly while using less memory. Certain parts of the model are still kept in single-precision floating point (`float32`) for numeric stability, but this does not affect the accuracy of the model.
    - **Example**: Let's say we have a neural network with 100 million parameters that we want to train on a dataset of images. If we train the network using single-precision floating point numbers (float32), the training will take about 10 days to complete.

        However, if we use mixed precision training, we can reduce the training time to about 5 days. This is because mixed precision training uses half-precision floating point numbers (float16) for most operations, which takes up half as much memory and is twice as fast as float32.

        The only parts of the network that are still kept in float32 are the weights and activations of the first few layers. This is because these layers are more sensitive to numerical errors, and using float16 for these layers could lead to a loss of accuracy.

        However, even with these few layers in float32, the overall accuracy of the network is still 99%. This shows that mixed precision training can be a very effective way to speed up the training of neural networks without sacrificing accuracy.
    
    - **Numeric stability**: Measure of how well a model's quality is maintained when using a lower-precision floating point data type (`float16` or `bfloat16`) instead of a higher-precision data type (`float32`). An operation is considered "numerically unstable" in float16 or bfloat16 if it causes the model to have worse evaluation accuracy or other metrics compared to running the operation in `float32`.
- Modern accelerators can take advantage of lower-precision data types, such as `float16` and `bfloat16`, which take up half as much memory as float32 and can be processed more quickly.
- Automatic Mixed Precision (AMP) is a technique that uses lower precision (such as `float16` or `bfloat16`) for operations that don't require the full precision of `float32`. This can speed up training without sacrificing accuracy.
- Mixed precision is a technique that can reduce the runtime and memory footprint of your network by matching each operation to its appropriate data type.

## Under-the-hood
- Using lower-precision floating-point (`float16` and `bfloat16`) formats can speed up machine learning operations on NVIDIA GPUs and TPUs. However, it is important to use `float32` for some variables and computations to ensure that the model trains to the same quality. This is because lower-precision floating-point formats can introduce numerical errors, which can affect the accuracy of the model.
- NVIDIA GPUs with Tensor Cores can achieve significant performance gains for `fp16` matrix operations, as these cores are specifically designed for this task.
- Using tensor cores in PyTorch used to be difficult, requiring manual writing of reduced precision operations into models. However, the `[torch.cuda.amp]()` API automates this process, making it possible to implement mixed precision training in just five lines of code. This can significantly speed up training time without sacrificing accuracy.

### How Mixed Precision Works
- To understand mixed precision training, we need to first understand floating point numbers. 
- In computer engineering, decimal numbers are typically represented as floating-point numbers. Floating-point numbers have a limited precision, but they can represent a wide range of values. This is a trade-off between precision and size.
    -  The number π cannot be represented exactly as a floating-point number, but it can be represented with a high degree of precision. This is sufficient for most engineering applications.
- Building upon the background of precision(Discussed Earlier), the IEEE 754 standard, which is the technical standard for floating point numbers, sets the following standards:
    - **Precision**: The number of digits that can be represented accurately in a floating point number. This is typically between 6 and 15 significant digits for single-precision and double-precision numbers, respectively.
    - **Range**: The set of all possible values that a floating point number can represent. This ranges from very small numbers (such as $10^{-38}$) to very large numbers (such as $10^{38}$).
    -**Rounding mode**: How floating point numbers are rounded when they are converted to integers. There are several different rounding modes, such as round to nearest, round down, and round up.
    - `fp64`, also known as double-precision or “double”, max rounding error of $~2^−52$
    - `fp32`, also known as single-precision or “single”, max rounding error of $~2^−23$.
    - `fp16` also known as half-precision or “half”, max rounding error of $~2^−10$
- PyTorch, which is more memory-sensitive than Python, uses `fp32` as its default dtype instead of `fp64`, which is the default float type in Python.
> Note: Mixed precision training is a technique that uses half-precision floating point numbers (`fp16`) instead of single-precision floating point numbers (`fp32`) to reduce the training time of deep learning models.
- The tricky part is to do it without compromising accuracy.
- Using smaller floating point numbers can lead to rounding errors that are large enough to cause underflow. This is a problem because many gradient update values during backpropagation are very small but not zero. Rounding errors can accumulate during backpropagation, turning these values into zeroes or NaNs. This can lead to inaccurate gradient updates and prevent the network from converging.
- The researchers "[Mixed Precision Training](https://arxiv.org/pdf/1710.03740.pdf)" found that using `fp16` "half-precision" floating point numbers for all computations can lose information, as it cannot represent gradient updates smaller than "$2^{-24}$" value. This information loss can affect the accuracy of the model, as around 5% of all gradient updates made by their example network were smaller than this threshold.
- Mixed precision training is a technique that uses `fp16` to speed up model training without sacrificing accuracy. It does this by combining three different techniques:
    - Maintain two copies of the weights matrix:
        -  The master copy of the weights matrix is stored in `fp32`. This is the copy that is used to calculate the loss function and to update the weights.
        - The `fp16` copy of the weights matrix is used for all other computations. This helps to speed up training by reducing the amount of memory required.
    - Use different vector operations for different parts of the network:
        - Convolutions are generally safe to run in `fp16`. This is because convolutions only involve multiplying small matrices together, which can be done accurately in `fp16`.
        - Matrix multiplications are not as safe to run in `fp16`. This is because matrix multiplications involve multiplying large matrices together, which can lead to rounding errors in `fp16`.
        - By using convolutions in `fp16` and matrix multiplications in fp32, we can improve accuracy while still using mixed precision.
    - Use loss scaling:
        - Loss scaling is a technique for multiplying the loss function output by a scalar number before performing backpropagation. This increases the magnitude of the gradient updates, which can help to prevent them from becoming too small.
        - A good loss scaling factor to start with is `8` "Suggested by paper". If the model is diverging, you can try increasing the loss scaling factor. However, if the loss scaling factor is too large, it can cause the model to diverge in the other direction.
- The authors used a combination of three techniques to train a variety of networks much faster than traditional methods. For benchmarks, please see the [paper](https://arxiv.org/pdf/1710.03740.pdf).

### How Tensor Cores Actually Works
- Mixed precision training (an `fp16` matrix is half the size of a `fp32` one) can reduce the memory requirements for deep learning models, but it can only speed up training if the GPU has special hardware support for half-precision operations. Tensor cores in recent NVIDIA GPUs provide this support, and can significantly speed up mixed precision training.
- Tensor cores are a type of processor that is specifically designed to perform a single operation very quickly: multiplying two 4x4 matrices of floating-point numbers in half precision (`fp16`) and adding the result to a third 4x4 matrix of floating-point numbers in either half precision or single precision (`fp32`). This operation is called a "fused multiply add".
- Tensor cores are a type of processor that can be used to accelerate matrix multiplication operations in half precision. This makes them ideal for accelerating backpropagation, which is a computationally intensive process that is used to train neural networks.

> Note: Tensor cores are only useful for accelerating matrix multiplication operations if the input matrices are in half precision. If you are training a neural network on a GPU with tensor cores and not using mixed precision training, you are wasting the potential of the GPU because the tensor cores will not be used.
- Tensor cores are a type of processor that was introduced in the Volta architecture in late 2017. They have been improved in the Turing and Ampere architectures, and are now available on the [V100](https://www.nvidia.com/en-us/data-center/v100/) and T4 GPUs. The V100 has 5120 CUDA cores and 600 tensor cores, while the T4 has 2560 CUDA cores and 320 tensor cores.Tensor cores can be used to accelerate matrix multiplication operations in half precision, which can significantly improve the performance of deep learning workloads.
> Although all versions of CUDA 7.0 or higher support tensor core operations, early implementations of tensor cores in CUDA were buggy. It is recommended to use CUDA 10.0 or higher for the best performance and stability when using tensor cores.

## How PyTorch Automatic Mixed Precision Works
- Now that we have covered the important background information, we can finally start exploring the new PyTorch `amp` API.
- Mixed precision training has been possible for a long time, but it required manual intervention to convert parts of the network to fp16 and implement loss scaling. Automatic mixed precision training automates these steps, making it as simple as adding two new API primitives to your training script: `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast`.
- Here is a code snippet that shows how to use mixed precision training in the training loop of a neural network. The comment "# NEW" marks the lines of code that have been added to enable mixed precision training.

```python
self.train()
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, self.max_lr,
    cycle_momentum=False,
    epochs=self.n_epochs,
    steps_per_epoch=int(np.ceil(len(X) / self.batch_size)),
)
batches = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y),
    batch_size=self.batch_size, shuffle=True
)

# NEW
scaler = torch.cuda.amp.GradScaler()

for epoch in range(self.n_epochs):
    for i, (X_batch, y_batch) in enumerate(batches):
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        optimizer.zero_grad()

        # NEW
        with torch.cuda.amp.autocast():
            y_pred = model(X_batch).squeeze()
            loss = self.loss_fn(y_pred, y_batch)

        # NEW
        scaler.scale(loss).backward()
        lv = loss.detach().cpu().numpy()
        if i % 100 == 0:
            print(f"Epoch {epoch + 1}/{self.n_epochs}; Batch {i}; Loss {lv}")

        # NEW
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
```

### Loss/Gradient Scaling
- Gradients with small magnitudes in `float16` may underflow and be lost, so it is important to use gradient scaling to prevent this.
- When training neural networks with float16, gradients with small magnitudes may underflow and be lost. Gradient scaling can be used to prevent this by multiplying the network's loss by a scale factor, which increases the magnitude of the gradients. This ensures that the gradients are large enough to be representable in float16 and that they are not lost.
- The PyTorch `GradScaler` object is a tool that can be used to prevent gradients from rounding down to 0 during training with mixed precision. It does this by multiplying the network's loss by a scale factor, which ensures that the gradients are large enough to be representable in float16. The optimal scale factor is one that is high enough to retain very small gradients, but not so high that it causes very large gradients to round up to `inf`.
- The optimal loss multiplier for mixed precision training is difficult to find because it varies depending on the network architecture, the dataset, and the learning rate. Additionally, the optimal multiplier can change over time as the gradients become smaller during training.
- To prevent gradient updates containing inf values, PyTorch uses exponential backoff. `GradScalar` starts with a small loss multiplier, which it doubles every so often. If GradScalar encounters a gradient update with inf values, it will discard the batch, divide the loss multiplier by 2, and start over.
- PyTorch uses an algorithm similar to TCP congestion control to approximate the appropriate loss multiplier over time. The algorithm steps the loss multiplier up and down, increasing it until it encounters a problem and then decreasing it until the problem is resolved. The exact numbers used by the algorithm are configurable.

```python
torch.cuda.amp.GradScaler(
    init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
    growth_interval=2000, enabled=True
)
```
- `GradScalar` needs to control the gradient update calculations and the optimizer to implement its behavior. This is why the `loss.backwards()` function is replaced with `scaler.scale(loss).backwards()`, and the `optimizer.step()` function is replaced with `scaler.step(optimizer)`.
- `GradScalar` can prevent gradient updates from overflowing, but it cannot prevent them from underflowing. This is because 0 is a legitimate value, while `inf` is not. If you pick a small `init_scale` and a large `growth_interval`, your network may underflow and diverge before GradScalar can intervene. To prevent this, it is a good idea to pick a large `init_scale`. The default `init_scale` of 65536 (2<sup>16</sup>) is a good choice.
- `GradScalar` needs to be saved to disk along with the model weights when checkpointing a model. This is because GradScalar is a stateful object that maintains its state across training iterations. The `state_dict()` and `load_state_dict()` object methods can be used to save and load the GradScalar state.
- To prevent the scale factor from interfering with the learning rate, it is necessary to unscale the `.grad` attribute of each parameter in PyTorch before the optimizer updates the parameters.

### `autocast` Context Manager
- The `torch.cuda.amp.autocast` context manager is a powerful tool for improving the performance of PyTorch models. It automatically casts operations to fp16, which can significantly speed up training without sacrificing accuracy. However, not all operations are safe to run in fp16, so it is important to check the amp [module documentation](https://pytorch.org/docs/master/amp.html#autocast-op-reference) for a list of supported operations.
- The list of operations that autocast can cast to fp16 is dominated by matrix multiplication and convolutions. The simple linear function is also supported.
- The operations listed above are safe to use in `FP16`, and they have up-casting rules to ensure that they are not affected by a mixture of `FP16` and `FP32` inputs. These operations include two other fundamental linear algebraic operations: matrix/vector dot products and vector cross products.
- The following operations are not safe to use in `FP16`: logarithms, exponents, trigonometric functions, normal functions, discrete functions, and large sums. These operations must be performed in `FP32` to avoid errors.
- Convolutional layers are the most likely layers to benefit from autocasting, as they rely on safe FP16 operations. Activation functions, on the other hand, may not benefit as much from autocasting, as they often use unsafe FP16 operations.
- To enable autocasting, you can simply wrap the forward pass of your model in the autocast context manager. This will cause all of the operations in the forward pass to be cast to `FP16`, except for those that are not safe to use in `FP16`.

```python
with torch.cuda.amp.autocast():
    y_pred = model(X_batch).squeeze()
    loss = self.loss_fn(y_pred, y_batch)
```
- When you wrap the forward pass of your model in the autocast context manager, autocasting will be automatically enabled on the backward pass as well(Example: `loss.backwards()`). This means that you only need to call autocast once, regardless of whether you are using the forward pass or the backward pass.
- Autocasting is a powerful tool that can help you to improve the performance of your PyTorch models. However, it is important to follow best practices for using PyTorch, such as avoiding in-place operations, to ensure that autocasting works correctly.


### Multiple GPUs
- Autocasting is compatible with the multi-GPU DistributedDataParallel API and the DataParallel multi-GPU API. With `DistributedDataParallel`, you need to use one process per GPU. With `DataParallel`, you need to make a [small adjustment](https://pytorch.org/docs/master/notes/amp_examples.html#dataparallel-in-a-single-process).
- The "Working with multiple GPUs" section of the [Automatic Mixed Precision Examples](https://pytorch.org/docs/master/notes/amp_examples.html#working-with-multiple-gpus) page in the PyTorch documentation is a good resource for learning how to use autocasting with multiple GPUs. The most important thing to remember is that you need to use torch.nn.BCEWithLogitsLoss instead of torch.nn.BCELoss if you want to get accurate results. [Source:prefer binary cross entropy with logits over binary cross entropy](https://pytorch.org/docs/master/amp.html#prefer-binary-cross-entropy-with-logits-over-binary-cross-entropy)


## Mixed Precision with TensorFlow
- The [TensorFlow: Mixed Precision](https://www.tensorflow.org/guide/mixed_precision) guide provides instructions on how to use mixed precision to train your TensorFlow models faster and with less memory.

## Performance Benchmarks
- Three neural networks were benchmarked in real-world settings using `V100`s (last-gen tensor cores) and `T4`s (current-gen tensor cores), the Spell API(cloud-based service that provides spell checking and grammar checking for text) on AWS EC2 instances (`p3.2xlarge` and `g4dn.xlarge` respectively), and a recent PyTorch build with CUDA 10.0. The benchmarks evaluated the performance of the networks with and without automatic mixed precision.
- The models trained using mixed precision and vanilla training converged to the same accuracy. The networks that were trained were:

| Model | Type | Dataset | Code |
| ------ | ---- | ------ | :----:|
| Feedforward | Feedforward neural network | [Rossman Store Samples](https://www.kaggle.com/c/rossmann-store-sales) competition on Kaggle | [🔗](https://github.com/spellml/feedforward-rossman)|
| UNet | Image segmentation network | [Segmented Bob Ross Images corpus](https://www.kaggle.com/datasets/residentmario/segmented-bob-ross-images)|[🔗](https://github.com/spellml/unet-bob-ross)|
| BERT | Natural language processing [transformer](https://jalammar.github.io/illustrated-transformer/) model([bert-base-uncased](https://huggingface.co/bert-base-uncased))	 |[Twitter Sentiment Extraction](https://www.kaggle.com/c/tweet-sentiment-extraction) competition on Kaggle	|[🔗](hhttps://github.com/spellml/tweet-sentiment-extraction)|

Thre results:
- Observation from results:
    -   Mixed precision training does not provide any benefits for the feedforward network because it is too small.
    - UNet, a medium-sized convolutional model with 7.7 million parameters, sees significant benefits from mixed precision training, especially on T4 GPUs, where it can save up to 30% of training time.
    - BERT is a large model that benefits greatly from mixed precision training. Automatic mixed precision can cut training time for BERT on Volta or Turing GPUs by up to 60%.

> The benefits of mixed precision training are immense, and it can be implemented with just a few lines of code in model training script. Given the potential to save up to 60% of training time with mixed precision training, it should be a top priority for performance optimization in model training scripts.

### What about Memory?
- Mixed precision training can be beneficial for memory usage, as `fp16` matrices are half the size of `fp32` matrices. This can be helpful for training larger models or for using larger batch sizes.
- PyTorch reserves GPU memory at the start of training to protect the training script from other processes that may try to use up too much memory and cause it to crash.
- Enabling mixed precision training can free up GPU memory, which can allow you to train larger models or use larger batch sizes.
- Both UNet and BERT benefited from mixed precision training, but UNet benefited more. The reason for this is not clear to me, as PyTorch memory allocation behavior is not well-understood.

# Conclusion
- The [PyTorch official website](https://pytorch.org/tutorials/#model-optimization) has tutorials that can help you get started with quantizing your models in PyTorch.
- If you are working with sequence data, start with…
    - [Dynamic quantization for LSTM](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html) or 
    [Dynamic quantization for BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
- If you are working with image data, you can start by learning about [transfer learning with quantization](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html). Once you have a good understanding of that, you can explore s[tatic post-training quantization](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html).
    - If you are not satisfied with the accuracy of your model after post-training quantization, you can try [quantization aware training](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) to improve accuracy.
- Deep learning researchers have developed model-specific methods to distill large models into smaller, faster models with similar performance.
- These distilled models can be used to gain performance without having to train a large model from scratch.
- In NLP, [HuggingFace](https://huggingface.co/) provides pre-trained distilled models such as DistilBert and TinyBert.
- In computer vision, [Facebook Research's d2go](https://github.com/facebookresearch/d2go) provides pre-trained mobile-ready models, some of which are distilled using DeiT methods.- 
- The paper "[Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962)" recommends that the best approach for training BERT architectures is to use a pre-trained model with a small number of parameters, and then fine-tune the model on a specific task. This approach was found to be more effective than training a BERT model from scratch, or fine-tuning a large BERT model on a specific task.
- One of the biggest advantages of the Pre-trained Distillation (PD) method is that it can be used with any NLP model architecture. This makes it a very versatile and powerful tool for training compact NLP models. If you are planning on using a compact NLP model in practice, I recommend reading the paper, especially section 6, which provides more details on the PD method.
- Automatic mixed precision training is a new feature that can speed up larger-scale model training jobs on recent NVIDIA GPUs by up to 60%. It is easy to use and does not require any changes to the model code.
- Automatic mixed precision training has been around for a while, but it was not easy to use for the average user because it required manual configuration. This has changed with the introduction of a native PyTorch API, which makes it much easier to use.
- The best place to learn more about mixed precision training is from the official PyTorch documentation. The [automatic mixed precision package](https://pytorch.org/docs/master/amp.html) and [automatic mixed precision examples pages](https://pytorch.org/docs/master/notes/amp_examples.html) are a great place to start.

# Key Takeaways
- The `torch.cuda.amp` mixed-precision training module can deliver significant speed improvements of up to 50-60% for large model training jobs, with only a few lines of code needed to be changed.

# Use-case
- Automatic Mixed Precision (AMP) is a technique that uses lower precision data types to reduce training time, while quantization is a technique that uses lower precision data types to reduce inference time.
- To learn how to use AMP in [PyTorch, you can refer to the PyTorch Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html)

# How Model Compression Techniques Reducing the cost? [Visual Summary]
- Not too long ago, the biggest Machine Learning models most people would deal with, merely reached a few GB in memory size. Now, every new generative model coming out is between 100B and 1T parameters. To get a sense of the scale, one float parameter that's 32 bits or 4 bytes, so those new models scale between 400 GB to 4 TB in memory, each running on expensive hardware. Because of the massive scale increase, there has been quite a bit of research to reduce the model size while keeping performance up.
- Model pruning is about removing unimportant weights from the network. The game is to understand what “important” means in that context. A typical approach is to measure the impact to the loss function of each weight. This can be done easily by looking at the gradient and second order derivative of the loss. Another way to do it is to use L1 or L2 regularization and get rid of the low magnitude weights. Removing whole neurons, layers or filters is called “structured pruning” and is more efficient when it comes to inference speed.
- Low-rank decomposition comes from the fact that neural network weight matrices can be approximated by products of low-dimension matrices. A $N×N$
matrix can be approximately decomposed into a product of $2N×1$ matrices. That’s a $O(N^2)−>O(N)$ space complexity gain
- Knowledge distillation is about transferring knowledge from one model to another. Typically from a large model to a smaller one. When the student model learns to produce similar output responses, that is response-based distillation. When the student model learns to reproduce similar intermediate layers, it is called feature-based distillation. When the student model learns to reproduce the interaction between layers, it is called relation-based distillation.
- Lightweight model design is about using knowledge from empirical results to design more efficient architectures. That is probably one of the most used methods in LLM research.
- The image below ([source](https://newsletter.theaiedge.io/p/the-aiedge-model-compression-techniques)) provides a concise and visually appealing overview of some of the methods 

# Inference Optimizations
- Credits for this section go to [Sebastian Raschka](https://www.linkedin.com/in/sebastianraschka/).
- Here are five ways to optimize deep neural network models for faster inference. These techniques don’t change the model architecture.
    - Parallelization
    - Vectorization
    - Loop tiling
    - Operator fusion
    - Quantization

# On-Device Privacy
| Aspect | Description |
| -------- | --------- |
| On-Device Privacy (Edge Computing) | Processing data directly on user devices (e.g., smartphones) instead of sending it to central servers. Enhances privacy and security by keeping data on the device, reducing exposure risks during transit or from compromised servers. | 
| NLP and LLM Systems in On-Device Processing | All interactions, including analysis and response generation in Natural Language Processing (NLP) and Language Model (LLM) systems, occur locally on the device. Important for maintaining privacy in conversational AI and lowering network-related latency. | 
| Benefits of On-Device Processing	| 1. Enhanced Privacy<br>2. Improved Security<br>3. Reduced Data Exposure Risks<br>4. Lower Latency for Smoother User Experience | 
| Solutions to Efficiency Challenges | Advances in model compression techniques:<br>- Pruning: Removing less important model components.<br>- Quantization: Reducing precision of numerical values.<br>- Knowledge Distillation: Transferring knowledge from a large model to a smaller one.<br>These techniques enable effective deployment of smaller models on devices.|

# Differential Privacy
| Aspect | Description | 
| ------- | ----------- |
| Differential Privacy	| A mathematical framework to measure individual privacy within a dataset. Involves adding random noise to data to hinder identification of individuals while preserving overall data patterns. | 
| Role in NLP and LLMs | In Natural Language Processing (NLP) and Language Model (LLM) applications, differential privacy ensures model outputs do not expose sensitive training data. Prevents models from generating text that might link back to specific individuals in the training data. | 
| Implementation Challenges in LLMs	 | While the principle of differential privacy is strong, applying it to complex models like LLMs is intricate. Striking a balance between privacy-preserving noise and model utility is crucial. | 
| Key Considerations | 1. Privacy vs. Utility: Finding the right amount of noise to protect privacy without rendering the model's output useless.<br>2. Complex Model Dynamics: Handling intricate interactions within LLMs to maintain data patterns.<br>3. Noise Addition: Incorporating noise without compromising the quality of language generation. | 
| Benefits of Differential Privacy	| 1. Privacy Preservation: Shields sensitive data in the training set, safeguarding individual information.<br>2. Legal and Ethical Compliance: Supports compliance with privacy regulations by minimizing data exposure.<br>3. Trust Building: Users are more inclined to use systems that prioritize their privacy and data security.| 
| Challenges in Implementation	| 1. Optimal Noise Level: Determining the appropriate noise magnitude for effective privacy.<br>2. Trade-off with Utility: Avoiding excessive noise that hampers model usefulness.<br>3. Performance Impact: Addressing potential model performance reduction due to noise introduction. | 

# Federated Learning
| Aspect | Description|
| ------ | ---------- |
| Federated Learning | A machine learning technique training a model across multiple devices or servers while keeping data localized. Each device learns a local model, periodically updating a global model. Raw data remains on the original device, preserving privacy. | 
| Role in NLP and Conversational AI	| In NLP and conversational AI, federated learning enables models to learn from diverse data sources without compromising privacy. For instance, a conversational AI learns from various devices, comprehending different contexts and dialects, while never accessing specific conversation data.|
| Benefits | 1. Privacy-Preserving Learning: Raw data remains on devices, enhancing user data privacy.<br>2. Diverse Data Insights: Models gain understanding from varied sources, broadening their knowledge and linguistic capabilities.<br>3. Decentralized Training: Training on local devices reduces the need to transfer large datasets to a central server.|
| Implementation Challenges	| 1. Efficient Aggregation: Coordinating and merging local model updates securely and efficiently.<br>2. Heterogeneous Devices: Handling different device capabilities and computational resources during the learning process.<br>3. Network Variability: Managing varying network connectivity affecting data synchronization.|
| Use Case Example	| 	Federated learning allows a conversational AI to learn from interactions on numerous devices, enabling it to understand diverse conversations, yet preserving the privacy of each conversation's raw data. |
| Considerations | 1. Model Drift: Local models may evolve differently, necessitating strategies to maintain global model performance.<br>2. Security: Ensuring secure communication during aggregation to prevent unauthorized access or tampering.|

# Low-rank Decomposition
- Low-rank decomposition can be used to compress neural networks by representing the weight matrices as the product of smaller matrices, which requires less storage space and computational resources.
- Low-rank decomposition can be used to compress large matrices (N * N) in neural networks by representing them as the product of two smaller matrices each of size $N * 1$. This can significantly reduce the space complexity of the matrices, from quadratic $O(N^2)$ to linear O(N), which can lead to significant improvements in computational efficiency.


# References
- [PyTorch official documentation: Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [PyTorch official documentation: Advanced Quantization in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [PyTorch official documentation: Quantization](https://pytorch.org/docs/stable/quantization.html)
- [CoreML Tools documentation: Quantization](https://coremltools.readme.io/docs/quantization)
- [PyTorch: Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch: Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [TensorFlow: Pruning Tutorial](https://www.tensorflow.org/model_optimization/guide/pruning/)
- [Pytorch Model Optimization: Automatic Mixed Precision vs Quantization](https://stackoverflow.com/questions/70503585/pytorch-model-optimization-automatic-mixed-precision-vs-quantization)