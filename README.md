# Custom-Large-Language-Model
Allows for domain-specific knowledge, customized responses, enhanced creativity, improved conversational abilities, data privacy, local deployment, and research opportunities. It offers tailored and specialized AI applications by fine-tuning the model to specific preferences, guidelines, and subject matter expertise.

# Prompt Engineering

Prompt engineering is a technique that can be used to improve the performance of LLMs on specific tasks. It involves crafting prompts that help the LLM to generate the desired output. This can be done by providing the model with additional information, such as examples of the desired output, or using specific language the model is likely to understand. 

Prompt Engineering can be powerful tool, but it is important to note that it is not a silver bullet. LLMs can still generate incorrect or unexpected output even with the best prompts. As a result, it is important to test the output of LLMs carefully before using them in production. 

# Fine Tuning
In other hand, fine-tuing is adapting a pre-trained LLM to a specific task or domain by training it further on a smaller, task-specific dataset. This is done by adjusting the model's weights and parameters to minimize the loss function and improve its performance on the task.

Fine-tuning can be more effective way to improve the performance of LLMs on specific tasks that prompt engineering. However, it is also more time-consuming and expensive. As a result, it is important to consider the cost and time involved in fine-tuning before deciding whether to use it. 

# When to perform

Fine-tuning is typically needed when the task is new or challenging or when the desired output is highly specific. In these cases, prompt engineering may not be able to provide the model with enough information to generate the desired output. 

Prompt engineering is typically sufficient when the task is well-defined and the desired output could be more specific. In these cases, prompt engineering can provide the model with the information it needs to generate the desired output. 


# What I am learning

After immersing myself in the recent GenAI text-based language model hype for nearly a month, I have made several observations about its performance on my specific tasks.

Please note that these observations are subjective and specific to my own experiences, and your conclusions may differ.

- We need a minimum of 7B parameter models (<7B) for optimal natural language understanding performance. Models with fewer parameters result in a significant decrease in performance. However, using models with more than 7 billion parameters requires a GPU with greater than 24GB VRAM (>24GB).
- Benchmarks can be tricky as different LLMs perform better or worse depending on the task. It is crucial to find the model that works best for your specific use case. In my experience, MPT-7B is still the superior choice compared to Falcon-7B.
- Prompts change with each model iteration. Therefore, multiple reworks are necessary to adapt to these changes. While there are potential solutions, their effectiveness is still being evaluated.
- For fine-tuning, you need at least one GPU with greater than 24GB VRAM (>24GB). A GPU with 32GB or 40GB VRAM is recommended.
- Fine-tuning only the last few layers to speed up LLM training/finetuning may not yield satisfactory results. I have tried this approach, but it didn't work well.
- Loading 8-bit or 4-bit models can save VRAM. For a 7B model, instead of requiring 16GB, it takes approximately 10GB or less than 6GB, respectively. However, this reduction in VRAM usage comes at the cost of significantly decreased inference speed. It may also result in lower performance in text understanding tasks.
- Those who are exploring LLM applications for their companies should be aware of licensing considerations. Training a model with another model as a reference and requiring original weights is not advisable for commercial settings.
- There are three major types of LLMs: basic (like GPT-2/3), chat-enabled, and instruction-enabled. Most of the time, basic models are not usable as they are and require fine-tuning. Chat versions tend to be the best, but they are often not open-source.
- Not every problem needs to be solved with LLMs. Avoid forcing a solution around LLMs. Similar to the situation with deep reinforcement learning in the past, it is important to find the most appropriate approach.
- I have tried but didn't use langchains and vector-dbs. I never needed them. Simple Python, embeddings, and efficient dot product operations worked well for me.
- LLMs do not need to have complete world knowledge. Humans also don't possess comprehensive knowledge but can adapt. LLMs only need to know how to utilize the available knowledge. It might be possible to create smaller models by separating the knowledge component.
- The next wave of innovation might involve simulating "thoughts" before answering, rather than simply predicting one word after another. This approach could lead to significant advancements.

# LLM Training Frameworks

| Framework | Description | Resource |
|------ | ---------- | :--------- |
|Alpa| Alpa is a system for training and serving large-scale neural networks. | [🔗](https://github.com/alpa-projects/alpa)|
|DeepSpeed| DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective. | [🔗](https://github.com/microsoft/DeepSpeed)|
|Megatron-DeepSpeed| DeepSpeed version of NVIDIA's Megatron-LM that adds additional support for several features such as MoE model training, Curriculum Learning, 3D Parallelism, and others. | [🔗](https://github.com/microsoft/Megatron-DeepSpeed)|
|FairScale| FairScale is a PyTorch extension library for high performance and large scale training | [🔗](https://fairscale.readthedocs.io/en/latest/what_is_fairscale.html)|
|Megatron-LM| Ongoing research training transformer models at scale. | [🔗](https://github.com/NVIDIA/Megatron-LM)|
|Colossal-AI| Making large AI models cheaper, faster, and more accessible. | [🔗](hhttps://github.com/hpcaitech/ColossalAI)|
|BMTrain | Efficient Training for Big Models. | [🔗](https://github.com/OpenBMB/BMTrain)|
|Mesh Tensorflow | Mesh TensorFlow: Model Parallelism Made Easier. | [🔗](https://github.com/tensorflow/mesh)|
|maxtext | A simple, performant and scalable Jax LLM! | [🔗](https://github.com/google/maxtext)|
|gpt-neox | An implementation of model parallel autoregressive transformers on GPUs, based on the DeepSpeed library.| [🔗](https://github.com/EleutherAI/gpt-neox)|
|Trainer API| Provides an API for feature-complete training in PyTorch for most standard use cases | [🔗](https://huggingface.co/docs/transformers/main_classes/trainer)|
| Lighting | Deep learning framework to train, deploy, and ship AI products Lightning fast | [🔗](https://github.com/Lightning-AI/lightning)|

# Effective Deployment Strategies for Language Models
| Deployment Tools | Description | Resource |
|------ | ---------- | :--------- |
| SkyPilot | Run LLMs and batch jobs on any cloud. Get maximum cost savings, highest GPU availability, and managed execution -- all with a simple interface. | [🔗](https://github.com/skypilot-org/skypilot)|
|vLLM | A high-throughput and memory-efficient inference and serving engine for LLMs | [🔗](https://github.com/vllm-project/vllm)|
|Text Generation Inference | A Rust, Python and gRPC server for text generation inference. Used in production at HuggingFace to power LLMs api-inference widgets. | [🔗](https://github.com/huggingface/text-generation-inference)|
| Haystack | an open-source NLP framework that allows you to use LLMs and transformer-based models from Hugging Face, OpenAI and Cohere to interact with your own data. | [🔗](https://haystack.deepset.ai/)|
| Sidekick |  Data integration platform for LLMs. | [🔗](https://github.com/psychic-api/psychic)|
| LangChain |  Building applications with LLMs through composability | [🔗](https://github.com/hwchase17/langchain)|
| wechat-chatgpt | Use ChatGPT On Wechat via wechaty | [🔗](https://github.com/fuergaosi233/wechat-chatgpt)|
| promptfoo | Test your prompts. Evaluate and compare LLM outputs, catch regressions, and improve prompt quality. | [🔗](https://github.com/promptfoo/promptfoo)|

## will add more
