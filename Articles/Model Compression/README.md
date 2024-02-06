# Enhancing Model Compression: Inference and Training Optimization Strategies

- Are you familiar with the implementation of model compression techniques for both training and inference in various machine learning and deep learning models? If not then you will acquire knowledge about various model compression and optimization concepts applicable during both model training and inference. These concepts encompass quantization/binarization, pruning, knowledge distillation, mixed precision training, and quantization-aware training.
- The exponential increase in the number of parameters in machine learning and deep learning models have become larger and larger, the computational requirements for training and inferencing them have also increased exponentially. This can make them impractical for use in edge environments or for serving to customers at scale.
- This article highlights state-of-the-art techniques developed by researchers and practitioners to enhance the efficiency of machine learning and deep learning models, enabling them to run more swiftly and with reduced memory consumption on edge devices 

# How Model Compression Techniques Reducing the cost? [Visual Summary]
- Not too long ago, the biggest Machine Learning models most people would deal with, merely reached a few GB in memory size. Now, every new generative model coming out is between 100B and 1T parameters. To get a sense of the scale, one float parameter that's 32 bits or 4 bytes, so those new models scale between 400 GB to 4 TB in memory, each running on expensive hardware. Because of the massive scale increase, there has been quite a bit of research to reduce the model size while keeping performance up.
- Model pruning is about removing unimportant weights from the network. The game is to understand what “important” means in that context. A typical approach is to measure the impact to the loss function of each weight. This can be done easily by looking at the gradient and second order derivative of the loss. Another way to do it is to use L1 or L2 regularization and get rid of the low magnitude weights. Removing whole neurons, layers or filters is called “structured pruning” and is more efficient when it comes to inference speed.
- Low-rank decomposition comes from the fact that neural network weight matrices can be approximated by products of low-dimension matrices. A $N×N$
matrix can be approximately decomposed into a product of $2N×1$ matrices. That’s a $O(N^2)−>O(N)$ space complexity gain
- Knowledge distillation is about transferring knowledge from one model to another. Typically from a large model to a smaller one. When the student model learns to produce similar output responses, that is response-based distillation. When the student model learns to reproduce similar intermediate layers, it is called feature-based distillation. When the student model learns to reproduce the interaction between layers, it is called relation-based distillation.
- Lightweight model design is about using knowledge from empirical results to design more efficient architectures. That is probably one of the most used methods in LLM research.
- The image below ([source](https://newsletter.theaiedge.io/p/the-aiedge-model-compression-techniques)) provides a concise and visually appealing overview of some of the methods 
![Compression Techniques](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/c08149ee-c578-4043-9af6-e1da902bd930)

# References
- [PyTorch official documentation: Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [PyTorch official documentation: Advanced Quantization in PyTorch](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [PyTorch official documentation: Quantization](https://pytorch.org/docs/stable/quantization.html)
- [CoreML Tools documentation: Quantization](https://coremltools.readme.io/docs/quantization)
- [PyTorch: Quantization](https://pytorch.org/docs/stable/quantization.html)
- [PyTorch: Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
- [TensorFlow: Pruning Tutorial](https://www.tensorflow.org/model_optimization/guide/pruning/)
- [Pytorch Model Optimization: Automatic Mixed Precision vs Quantization](https://stackoverflow.com/questions/70503585/pytorch-model-optimization-automatic-mixed-precision-vs-quantization)
