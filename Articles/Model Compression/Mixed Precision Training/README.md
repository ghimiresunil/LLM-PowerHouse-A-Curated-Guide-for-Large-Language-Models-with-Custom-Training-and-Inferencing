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

### How Mixed Precision Actually Works
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

![mixed_precision](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/1316fdde-2bbc-49f9-99fd-aac4a462cbf6)

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
![How Tensor Cores Actually Works](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/c4958145-628a-4e22-9f06-61544eb02c81)
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
![image](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/ff9bff6e-2a18-4359-ba58-89fa8aee2ee0)

- The operations listed above are safe to use in `FP16`, and they have up-casting rules to ensure that they are not affected by a mixture of `FP16` and `FP32` inputs. These operations include two other fundamental linear algebraic operations: matrix/vector dot products and vector cross products.
![image](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/7bbd5347-b609-4a22-9d35-30a71ef383b5)

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

The results:
![result](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/60086032-8772-4e0b-9102-7f319217ffce)

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
![result_memory](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/06155d01-2f4f-4be6-8fe0-088bcbe59483)

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