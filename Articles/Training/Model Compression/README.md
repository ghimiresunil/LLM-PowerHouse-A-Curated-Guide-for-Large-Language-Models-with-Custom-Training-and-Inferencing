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

## Dynamic/Runtime Quantization

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

## Post-Training Static Quantization
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
## Static Quantization-aware Training (QAT)
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

# Device and Operator Support
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

# Integration in Torchvision
- PyTorch has made it easier to quantize popular models in [torchvision](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) by providing quantized versions of the models, a quantization API, and a quantization tutorial. This makes it possible to deploy quantized models to production with less effort.
    - Quantized versions of the models, which are pre-trained and ready to use.
    - Quantization-ready model definitions are provided so that you can quantize a model after training (post-training quantization) or quantize a model during training (quantization aware training).
    - Quantization aware training is a technique that can be used to improve the accuracy of any model, but we found it to be especially beneficial for Mobilenet
    - You can find a [tutorial](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html) on that demonstrates how to do transfer learning with quantization using a torchvision model.

# Choosing an Approach
- The decision of which scheme to use is dependent by a variety of factors.
    - **Model/Target requirements**: Some models are more sensitive to quantization than others, and may require quantization-aware training to maintain accuracy.
    - **Operator/Backend support**: There are backends that only work with fully quantized operators.
- The number of quantized operators available in PyTorch is currently limited, which may impact the choices you can make from the table below. This table, from PyTorch: Introduction to Quantization on PyTorch, provides some guidance.

# Performance Reults
- Quantization can reduce the model size by 4x and speed up inference by 2x to 3x, depending on the hardware platform and the model being benchmarked. The table below from the [PyTorch documentation on quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) provides some sample results of the technique.

# Accuracy Results
- The tables in [PyTorch's Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) document compare the accuracy of quantized models to floating-point models on the ImageNet, as well as we compared the F1 score of BERT on the GLUE benchmark for MRPC.

## Computer Vision Model Accuracy
## Speech and NLP Model Accuracy

# Conclusion
- The [PyTorch official website](https://pytorch.org/tutorials/#model-optimization) has tutorials that can help you get started with quantizing your models in PyTorch.
- If you are working with sequence data, start with…
    - [Dynamic quantization for LSTM](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html) or 
    [Dynamic quantization for BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
- If you are working with image data, you can start by learning about [transfer learning with quantization](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html). Once you have a good understanding of that, you can explore s[tatic post-training quantization](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html).
    - If you are not satisfied with the accuracy of your model after post-training quantization, you can try [quantization aware training](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html) to improve accuracy.

# Quantization in Other Frameworks: TensorFlow and CoreML
- PyTorch quantization may not work in all production environments, such as when converting a model to Apple's CoreML format, which requires 16-bit quantization. When deploying a model to an edge device, it is important to check that the device supports quantization. On Apple devices, the hardware already computes everything in `fp16`, so quantization is only useful for reducing the memory footprint of the model.
- TensorFlow uses a similar set of steps as above, but the examples are focused on TFLite. 
- The [post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) page explains static and dynamic quantization, and the QAT page provides more information about quantization aware training. The tradeoffs between PyTorch and TensorFlow for quantization are similar, but there are some features that are not compatible between the two frameworks.

# How Far Can We Go?
- Researchers have been working on binary neural networks [for years](https://arxiv.org/abs/1909.13863), as they offer the potential for extreme speedups with only a small loss in accuracy. These networks use only 1 bit of precision for their weights and activations, which can be much faster to compute than traditional neural networks. Binary neural networks are still mostly research projects, but [XNOR-Net++](https://arxiv.org/abs/1909.13863) is a notable exception as it has been implemented in PyTorch. This makes it a more usable idea for practical applications.

# Use-case
- Quantization is a technique that can be used to increase the speed of model inference by reducing the precision of the model's parameters. In contrast, please reed Mixed Precision Training, Automatic Mixed Precision (AMP) is a technique that can be used to reduce the training time of a model by using lower precision numbers during training.

# Further Reading
- [PyTorch official documentation: Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [PyTorch official documentation: Advanced Quantization in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [PyTorch official documentation: Quantization](https://pytorch.org/docs/stable/quantization.html)
- [CoreML Tools documentation: Quantization](https://coremltools.readme.io/docs/quantization)

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

# Distillation in Practice
- Knowledge distillation is a rapidly growing research field with applications in defending against adversarial attacks, transferring knowledge between models, and protecting privacy.
- In knowledge distillation, the student model learns from the teacher model by mimicking its predictions. Response-based distillation focuses on the final output layer of the teacher model, while feature-based and relation-based distillation focus on other parts of the teacher model.
- The different types of distillation, such as offline distillation (where the student model is trained after the teacher model), online distillation (where the student and teacher models are trained together), and self-distillation (where the teacher model has the same architecture as the student model), can make it difficult to track distillation in practice. A set of ad hoc model-specific techniques may be the best general recommendation.
- In Fact,
    - [Cho & Hariharan (2019)](https://arxiv.org/abs/1910.01348) found that knowledge distillation can be harmful when the student model is too small. They also found that knowledge distillation papers rarely use ImageNet and so often don't work well on difficult problems.
    - [Mirzadeh et al. (2019)](https://arxiv.org/abs/1902.03393) found that better teacher models don't always mean better distillation, and that the farther the student and teacher model's capacities are, the less effective distillation is.
    - A recent investigation by [Tang et al. (2021)](https://arxiv.org/pdf/2002.03532.pdf) supports these findings.
> Note: Knowledge distillation can be harmful when the student model is too small, and it is less effective when the student and teacher models have different capacities.
- In Summary, knowledge distillation is a powerful technique for improving the performance of small models, but it is more difficult to implement than quantization and pruning. However, it can be worth the effort if you need to achieve a high level of accuracy with a small model.

## Distillation As Semi-supervised Learning
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

# Fine Tuning | [What is Fine Tuning](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/tree/main/Articles/Training/Fine%20Tuning%20Models)
- After pruning a neural network, it is [standard practice](https://arxiv.org/pdf/2003.02389.pdf) to retrain the network. The best method is to reset the learning rate to its original value and start training from scratch. Optionally, you can also reset the weights of the unpruned parts of the network to their values earlier in training. This is essentially training the lottery ticket subnetwork that we have identified.
    - For example: let's say we have a neural network with 1000 weights. We use pruning to remove 90% of the weights, leaving us with 100 weights. We then retrain the network with a reset learning rate and the weights from the earlier training. This helps the lottery ticket subnetwork to learn how to perform the task at hand more effectively.

>  If you are interested in network pruning, you can start by using [TorchPruner](https://github.com/marcoancona/TorchPruner) or [Torch-Pruning](https://github.com/vainf/torch-pruning) to prune the network. Then, you can fine-tune the resulting network with learning rate rewinding. However, it is not always clear how to trim the rest of the network around the pruned part, especially for architectures with skip connections like ResNets.

# DeepSpeed and ZeRO-Offload
- [DeepSpeed](https://www.deepspeed.ai/) is a library that optimizes the training of large and extremely large models on GPUs. It does this by using smart parallelism and better caching, which can lead to significant speedups and memory savings. DeepSpeed is an extension to PyTorch, so it is easy to use with existing PyTorch code.

# Conclusion
- Deep learning researchers have developed model-specific methods to distill large models into smaller, faster models with similar performance.
- These distilled models can be used to gain performance without having to train a large model from scratch.
- In NLP, [HuggingFace](https://huggingface.co/) provides pre-trained distilled models such as DistilBert and TinyBert.
- In computer vision, [Facebook Research's d2go](https://github.com/facebookresearch/d2go) provides pre-trained mobile-ready models, some of which are distilled using DeiT methods.- 
- The paper "[Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962)" recommends that the best approach for training BERT architectures is to use a pre-trained model with a small number of parameters, and then fine-tune the model on a specific task. This approach was found to be more effective than training a BERT model from scratch, or fine-tuning a large BERT model on a specific task.
- One of the biggest advantages of the Pre-trained Distillation (PD) method is that it can be used with any NLP model architecture. This makes it a very versatile and powerful tool for training compact NLP models. If you are planning on using a compact NLP model in practice, I recommend reading the paper, especially section 6, which provides more details on the PD method.

# 