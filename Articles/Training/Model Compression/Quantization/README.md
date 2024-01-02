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
- The number of quantized operators available in PyTorch is currently limited, which may impact the choices you can make from the table below. This table, from [PyTorch: Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/), provides some guidance.
  
![choosing_an_approach](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/6fb70bc1-8315-4b61-9cfa-cc66a688c7c8)

## Performance Reults
- Quantization can reduce the model size by 4x and speed up inference by 2x to 3x, depending on the hardware platform and the model being benchmarked. The table below from the [PyTorch documentation on quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) provides some sample results of the technique.

![performance_result](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/cafd2132-0f7b-47e7-8425-2e56ae8cdabb)

## Accuracy Results
- The tables in [PyTorch's Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) document compare the accuracy of quantized models to floating-point models on the ImageNet, as well as we compared the F1 score of BERT on the GLUE benchmark for MRPC.

### Computer Vision Model Accuracy
![computer_vision_model_accuracy](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/bc12512e-3537-46b1-b946-812e99994934)

### Speech and NLP Model Accuracy
![speech_and_nlp_model_accuracy](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/7af7f87e-da68-46e5-84a5-f9286249b771)

## Quantization in Other Frameworks: TensorFlow and CoreML
- PyTorch quantization may not work in all production environments, such as when converting a model to Apple's CoreML format, which requires 16-bit quantization. When deploying a model to an edge device, it is important to check that the device supports quantization. On Apple devices, the hardware already computes everything in `fp16`, so quantization is only useful for reducing the memory footprint of the model.
- TensorFlow uses a similar set of steps as above, but the examples are focused on TFLite. 
- The [post-training quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) page explains static and dynamic quantization, and the QAT page provides more information about quantization aware training. The tradeoffs between PyTorch and TensorFlow for quantization are similar, but there are some features that are not compatible between the two frameworks.

## How Far Can We Go?
- Researchers have been working on binary neural networks [for years](https://arxiv.org/abs/1909.13863), as they offer the potential for extreme speedups with only a small loss in accuracy. These networks use only 1 bit of precision for their weights and activations, which can be much faster to compute than traditional neural networks. Binary neural networks are still mostly research projects, but [XNOR-Net++](https://arxiv.org/abs/1909.13863) is a notable exception as it has been implemented in PyTorch. This makes it a more usable idea for practical applications.

## Use-case
- Quantization is a technique that can be used to increase the speed of model inference by reducing the precision of the model's parameters. In contrast, please reed Mixed Precision Training, Automatic Mixed Precision (AMP) is a technique that can be used to reduce the training time of a model by using lower precision numbers during training.
