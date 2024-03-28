# Inference Optimizations
- Credits for this section go to [Sebastian Raschka](https://www.linkedin.com/in/sebastianraschka/).
- Here are five ways to optimize deep neural network models for faster inference. These techniques don’t change the model architecture.
    - Parallelization
    - Vectorization
    - Loop tiling
    - Operator fusion
    - Quantization
![Inference Optimizations](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/ccd14769-1652-410b-8862-ffce67e8dde6)

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

# Low-Rank Decomposition in Neural Networks

Low-rank decomposition is a technique used in neural networks to compress weight matrices, reducing storage requirements and computational resources.

| Application | Description |
|-------------|-------------|
| **Compression of Neural Networks** | By representing weight matrices as the product of smaller matrices, the storage space and computational resources required for neural networks can be reduced. This compression technique is particularly useful for deploying models in resource-constrained environments or for mobile applications where memory and computational power are limited. |
| **Compression of Large Matrices** | Large matrices in neural networks, typically of size N * N, can be compressed by representing them as the product of two smaller matrices, each of size N * 1. This decomposition reduces the space complexity of the matrices from quadratic O(N^2) to linear O(N), resulting in significant improvements in computational efficiency. This reduction in space complexity is particularly beneficial for tasks involving large-scale data processing, such as image and speech recognition. |



# Continuous Batching

Continuous Batching is a technique that maximizes GPU utilization. It involves:

| Technique            | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| Streamlining computation | By continuously feeding batches of data to the GPU.                                            |
| Reducing idle time  | Ensures that the GPU always has work to do.                                                        |
| Enhancing throughput | Minimizes the time spent waiting for I/O operations.                                           |

# Speculative Batching
Speculative Batching is a predictive approach that:

| Technique             | Description                                                                                          |
|-----------------------|------------------------------------------------------------------------------------------------------|
| Pre-executes tasks    | Based on the likelihood of their necessity.                                                          |
| Time-saving           | Preemptively processes data that will probably be needed.                                             |
| Increased efficiency  | Better resource utilization.                                                                         |
| Parallel validation   | Speculations can be run in parallel to validate.                                                      |

# Attention Mechanisms

Attention mechanisms have revolutionized the way neural networks process data. Let’s look at some sophisticated variants:

## Flash Attention 2

| Feature                | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| High-speed processing  | Designed for rapid computation.                                                                |
| Efficient memory usage | Optimizes the use of memory bandwidth.                                                         |
| Basis                  | "GPUs are good at computation rather than read and write".                                      |
| Performance optimization | Reduces reads and writes, recomputes, partial softmax calculates to compensate.                  |

## Multi-head Attention (MHA)

| Feature                | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| Parallel processing    | Processes information in parallel across different representation subspaces.                    |
| Richer representations | Captures various aspects of the data simultaneously.                                            |
| Separate K and V for each q | Each query has a separate key and value.                                                     |

## Multi-query Attention (MQA)

| Feature                | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| Multiple queries       | Handles several queries in one go.                                                             |
| Enhanced context capture | Allows the model to consider multiple perspectives at once.                                      |
| Single K and V across all attention heads | Uses a single set of key and value across all attention heads to reduce memory burden.          |

## Group-query Attention (GQA)

| Feature                | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| Grouped processing     | Processes sets of queries together.                                                            |
| Improved relational understanding | Adept at understanding the relationships between different data points.                          |
| Grouped K and V across all attention heads | Uses grouped key and value across all attention heads to reduce memory burden.                  |

## Paged KV Cache for the Attention

| Feature                | Description                                                                                     |
|------------------------|-------------------------------------------------------------------------------------------------|
| Memory efficiency      | Uses a paged mechanism to store key-value pairs.                                                |
| Faster access          | Allows for quicker retrieval of relevant information during the attention process.               |
| Reduced memory fragmentation loss | Reduces memory fragmentation loss.                                                             |
| Disadvantage           | Makes the system memory-bound.                                                                  |

# KV Cache

## Prefill

| Technique      | Description                                                                                                 |
|----------------|-------------------------------------------------------------------------------------------------------------|
| Ready-to-use cache | Prefill prepares the KV cache with relevant data before it’s needed.                                        |
| Reduces latency    | By ensuring that data is immediately available when the model requires it.                                    |
| Batch prefilling   | Prefill can be done in batch to make most of compute available.                                                |

# Parallelism

Parallelism is key to scaling deep learning models. Here are three types:

## Tensor Parallelism

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Divides tensors    | It splits the computational workload across multiple GPUs.                                                  |
| Enables larger models | By distributing the memory requirements.                                                                     |

## Pipeline Parallelism

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Sequential stages  | It breaks the model into stages that are processed in sequence.                                              |
| Continuous workflow | Each GPU works on different stages simultaneously.                                                          |

## Data Parallelism

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Copies of the model | Each GPU has its own copy of the model.                                                                     |
| Synchronized learning | They all learn from different subsets of the data.                                                           |

# Optimizing

Optimization techniques refine the model’s performance and efficiency:

## Quantization

### GPTQ

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Gradient-based     | GPTQ uses gradients to quantize weights without significant loss of accuracy.                                |
| Balances performance | It maintains a balance between model size and effectiveness.                                                |

### AWQ

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Adaptive           | AWQ adjusts quantization levels based on the data distribution.                                              |
| Resource-efficient | It aims to use fewer bits where possible without compromising quality.                                       |

## Distillation

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Knowledge transfer | Distillation involves teaching a smaller model to mimic a larger one.                                        |
| Compact models     | The result is a more efficient model that retains much of the performance.                                   |

## Pruning

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Removes redundancy | Pruning cuts out unnecessary weights or neurons.                                                            |
| Streamlines models | It leaves a leaner, faster model that requires less computation.                                             |

# FP8

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Half-precision format | FP8 is a new floating-point format that uses only 8 bits.                                                  |
| Saves memory       | It significantly reduces the memory footprint of models.                                                     |
| Maintains precision | Despite its size, it’s designed to retain as much precision as possible.                                     |

# Greedy-search

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| One step at a time | Greedy-search selects the best option at each step without looking ahead.                                   |
| Fast and simple    | It’s a straightforward approach that can be very efficient.                                                  |

# Beam-search

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Explores multiple paths | Beam-search keeps track of several of the best options at each step.                                       |
| Balances breadth and depth | It’s more thorough than greedy-search but also more computationally intensive.                              |

# RoPE

| Technique          | Description                                                                                                 |
|--------------------|-------------------------------------------------------------------------------------------------------------|
| Rotary Positional Embedding | RoPE encodes the position information into the attention mechanism.                                      |
| Enhances understanding    | It helps the model better understand the order and relationship of elements in the sequence.                |
