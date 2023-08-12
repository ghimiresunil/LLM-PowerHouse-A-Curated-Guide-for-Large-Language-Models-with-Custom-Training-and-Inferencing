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






