# Parameter-Efficient Finetuning (PEFT)

PEFT stands for Parameter-Efficient Fine-Tuning. It is a technique used in Natural Language Processing (NLP) to improve the performance of pre-trained language models on specific downstream tasks. PEFT involves reusing the pre-trained model's parameters and fine-tuning them on a smaller dataset, which saves computational resources and time compared to training the entire model from scratch.

There are a number of different PEFT techniques, but they all share the same basic idea of fine-tuning only a small number of the model's parameters. This can be done by freezing some of the layers of the model, or by using a technique called weight decay to penalize the changes to certain parameters.

PEFT has been shown to be effective in a variety of NLP tasks, including question answering, text classification, and natural language inference. In some cases, PEFT can achieve performance that is comparable to full fine-tuning, while using significantly fewer resources.

Here are some of the benefits of using PEFT:
- Reduced computational resources: PEFT can save a significant amount of computational resources, as it does not require training the entire model from scratch. This makes it a viable option for tasks with limited computational resources, such as mobile devices or edge device
- Reduced training time: PEFT can also reduce the training time required for fine-tuning, as it only needs to fine-tune a small number of parameters. This can be a significant advantage for tasks where speed is important, such as real-time applications.
- Reduced risk of catastrophic forgetting: PEFT can help to reduce the risk of catastrophic forgetting, which is a phenomenon that occurs when a model is fine-tuned on a new task and forgets what it was originally trained on. This is because PEFT only fine-tunes a small number of parameters, which helps to preserve the knowledge that the model has already learned.
- Portability: PEFT methods enable users to obtain tiny checkpoints worth a few MBs compared to the large checkpoints of full fine-tuning. This makes the trained weights from PEFT approaches easy to deploy and use for multiple tasks without replacing the entire model.
- Better modeling performance: PEFT enables achieving comparable performance to full fine-tuning with only small number of trainable parameters.
- Less storage: majority of weights can be shared across different tasks

<hr/>
If you are working on an NLP task that requires a large language model, PEFT is a technique that you should consider. It can help you to save computational resources, reduce training time, and reduce the risk of catastrophic forgetting.
<hr/>

# Difference between fine-tuning and parameter-efficient fine-tuning
| Aspect | Fine Tuning | Parameter-efficient Fine-tuning|
| ------- | ---------- | ------------------------------- |
| Objective	| Enhance a pre-trained model's performance on a specific task using abundant data and significant computation power	| Improve a pre-trained model's performance on a specific task with constraints on data and computational resources|
| Training Data|	Extensive dataset (numerous examples)|	Limited dataset (few examples)|
|Training Duration|	Prolonged training duration compared to parameter-efficient approach|	Swift training process relative to traditional fine-tuning|
|Computational Demands|	Requires substantial computational resources|	Demands fewer computational resources|
|Model Parameters|	Completely retrains the entire model|	Modifies only a small fraction of the model parameters|
|Overfitting Risk|	Higher susceptibility to overfitting due to extensive model modification|	Reduced risk of overfitting as model modifications are more controlled|
|Training Performance|	Typically yields superior results compared to parameter-efficient method|	Performance not as high as traditional fine-tuning, but still satisfactory|
|Applicability|	Well-suited for scenarios with abundant data and ample computational resources|	Ideally suited for scenarios with limited resources or inadequate training data availability|

# Practical Use Case
- PEFT eliminates the necessity for 40 or 80GB A100 GPUs to utilize potent LLMs. Essentially, you can fine-tune LLMs with over 10 billion parameters for your specific task at no cost or using affordable consumer GPUs.
- By using PEFT techniques such as LoRA, particularly utilizing 4-bit quantized base models like QLoRA, it becomes feasible to fine-tune LLMs with over 10 billion parameters, which typically occupy 30-40GB of space, even on GPUs with 16GB memory. If acquiring a 16GB GPU or TPU exceeds your budget, occasional opportunities arise on Google Colab to access a 16GB VRAM Tesla T4 at no cost. It's important to periodically save your model checkpoints and reload them as needed, especially in case of a Colab disconnect or kernel crash.
- Even if you only have a few examples to train on, large language models (LLMs) are so powerful that they can still learn to perform well on a single task. With PEFT via LoRA, you can further reduce the amount of training data needed by training on a tiny fraction of the LLM's parameters (0.08%). This is possible because LoRA uses a technique called low-rank adaptation, which allows the LLM to learn to focus on the most important parameters for the task at hand. Even though the weights are stored as 4-bit integers, computations are still done at 16-bit precision, which ensures that the model's performance is not compromised.
- Although fine-tuning a model requires a significant amount of VRAM, using PEFT with a small batch size and little gradient accumulation can reduce the VRAM requirements while still using FP16 computation. In some cases, the performance of the fine-tuned model can be comparable to that of a model that was fine-tuned using 16-bit precision.
- Key Takeaway: You can use free compute to fine-tune powerful LLMs for your specific task. Use a model with fewer than 10 billion parameters, which is still a large model. You can also use quantization, PEFT, checkpointing, and a small training set to quickly fine-tune the model for your use case.

# Prompt Tuning 
- Prompt tuning is a simple yet effective method for conditioning frozen language models to perform specific downstream tasks. Unlike discrete text prompts, soft prompts are learned through backpropagation and can be tuned to incorporate signals from any number of labeled examples. This method was first introduced in the paper "[The Power of Scale for Parameter-Efficient Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf)" by Lester et al.
- Prompt tuning is a lightweight approach to adapting a pre-trained language model to new tasks. It only requires storing a small task-specific prompt for each task, and it allows the model to be used for multiple tasks without retraining.
- Prompt tuning consistently outperforms few-shot learning, and its performance becomes even more competitive as the scale of the dataset increases.
-  This approach is a promising way to use a single frozen model for efficient multi-task serving.
- Model tuning requires storing a separate copy of the pre-trained model for each task, while prompt tuning only requires storing a small task-specific prompt. This makes prompt tuning much more efficient, as it requires 11 billion fewer parameters per task. For example, a T5 “XXL” model with a prompt length of 5 tokens would only require 20,480 parameters.
- Model tuning requires storing a separate copy of the pre-trained model for each task, while prompt tuning only requires storing a small task-specific prompt, and enables mixed-task inference using the original pretrained model. Each instance of the fine-tuned, T5 “XXL” model requires 11 billion parameters per copy, while our tuned prompts require only 20,480 parameters per task, reducing the count by more than five orders of magnitude, assuming a prompt length of 5 tokens.

<hr/>

![PromptTuning](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/fba637df-f99e-4399-858a-288cecc562cb)

<hr/>

# Adapter
- Adapters are a parameter-efficient fine-tuning technique that achieves similar performance to tuning the top layers, while requiring as few as two orders of magnitude fewer parameters. They are small neural network modules that are inserted into LLMs to adapt them for executing new tasks. During fine-tuning, only the weights of the adapter are learned. I believe there might be some confusion regarding a few of the terms used here. 
- Adapter-based tuning is a technique that inserts new modules called "adapter modules" between the layers of a pre-trained language model. The full pre-trained model is frozen during fine-tuning, and only the adapter modules are optimized. This means that only a few parameters are introduced per task, resulting in "compact" models. 

I believe there might be some confusion regarding a few of the terms used here. I would be glad to provide an explanation for each term.

- **Adapters**: Adapters are small, trainable modules that are inserted between the layers of a pre-trained language model. They allow the model to be fine-tuned for a specific downstream task without having to retrain the entire model.
- **Parameter-efficient fine-tuning**: Parameter-efficient fine-tuning is a technique that uses a small number of parameters to fine-tune a pre-trained language model. This is in contrast to traditional fine-tuning, which uses a large number of parameters.
- **Two orders of magnitude**: Two orders of magnitude is a very large difference. It means that adapters can use as few as 100 times fewer parameters than traditional fine-tuning.
- **Similar performance**: Adapters have been shown to achieve similar performance to traditional fine-tuning, even though they use a much smaller number of parameters. This makes them a very efficient way to fine-tune pre-trained language models.
- **Small neural network modules**: Adapters are small, typically 1-2 layers deep, neural network modules. This makes them much smaller than the entire pre-trained language model, which can have millions or billions of parameters.
- **Inserted into LLMs**: Adapters are inserted into the pre-trained language model, typically between the attention and feed-forward layers. This allows the adapter to learn how to adapt the model's attention and feed-forward layers for a specific downstream task.
- **Weights of the adapter are learned**: During fine-tuning, only the weights of the adapter are learned. This means that the rest of the pre-trained language model is frozen, which prevents it from overfitting to the training data.
- **Adapter-based tuning**: Adapter-based tuning is a technique that uses adapter modules to fine-tune a pre-trained language model. Adapter modules are small, trainable modules that are inserted between the layers of the pre-trained model.
- **New modules**: Adapter modules are new modules that are added to the pre-trained model. These modules are typically 1-2 layers deep, and they are designed to adapt the pre-trained model for a specific downstream task.
- **Frozen**: The full pre-trained model is frozen during fine-tuning. This means that the weights of the pre-trained model are not updated. Only the weights of the adapter modules are updated.
- **Few parameters**: Only a few parameters are introduced per task when using adapter-based tuning. This is because the adapter modules are small, and they only have a few trainable parameters.
- **Compact models**: The resulting models are "compact" because they only have a few parameters. This makes them more efficient to train and deploy.

# Adapter Module
Let's look the implementation of the adapter module within the transformer architecture through three key aspects:
- The adapter module (right) first reduces the original d-dimensional features to a smaller m-dimensional vector, applies a nonlinear transformation, and then expands it back to d dimensions.
- The module has a skip-connection, which allows us to initialize the parameters of the projection layers to near-zero, effectively initializing the module to an identity function. This is important for stable fine-tuning, as it allows us to preserve the learning from pre-training.
- In a transformer block (left), the adapter is applied directly to the outputs of each of the layers (attention and feedforward).

# How to decide the value of $m$?

- The size of the adapter module (m) determines the number of optimizable parameters, which affects the trade-off between model size and performance.
- The original paper found that the performance of the model is consistent across different adapter sizes m. Therefore, a fixed adapter size can be used for all downstream tasks.
<hr/>

![adapter](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/4bddef4a-4a2e-46e1-9de3-3d833d8d8551)

<hr/>

# LOw Rank Adaptation (LoRA)

Are you struggling with high GPU memory costs while fine-tuning a Heavily Parameterized Large Language Model like LLM? Look no further!

The basic idea behind LoRa is:

```
Heavily Parameterized Large Language Models + Basic Linear Algebra Theorem = Save GPU memory!
```

- Adapters and prefix tuning are two fine-tuning techniques for multitask learning that have some potential downsides. 
    - Adapters can introduce inference latency, especially in online low batch size inference settings. Example: If you are using a multitask learning model to make predictions in real time, such as for a chatbot or a self-driving car, then adapters could introduce too much latency and make the model unusable.
    - Prefix tuning can reduce the model's usable sequence length. Example: If you are using a multitask learning model to process long sequences, such as for natural language processing or speech recognition, then prefix tuning could reduce the model's accuracy.
- LoRA, a PEFT technique that utilizes a straightforward concept of decomposing non-full rank matrices. LoRA, short for low rank adaptation of LLMs, focuses on parameter efficiency in fine-tuning.
- LoRA hypothesizes that “change in weights” during adaptation has a “low intrinsic rank”. $ΔW$ is non-full rank and so can be written as $ΔW=BA$
    - A matrix is rank-deficient if it has a rank less than the lesser of its number of rows and columns. For more, reference [Wikipedia: Rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra))
<hr/>

![1683516728965](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/834d8d79-83f0-41eb-8c8b-97dbf2b52609)

<hr/>

- The concept of "low intrinsic rank" is inspired by the idea of "low intrinsic dimensionality" in over-parameterized pre-trained models. This is also why fine-tuning only a part of the full model can yield good results, instead of fine-tuning the entire model.
- During training, the outputs from $W$ and $ΔW$ are added component wise, like so:
$$h = Wx + BAx$$
- The new matrices $B$ and $A$, which have a much smaller number of parameters combined than the full matrix, are now the only ones left to optimize.
- In summary, the pre-trained weights $W$ are not updated, and only the rank decomposition matrices $B$ and $A$ of the change in weight matrix are optimized.
- This approach yields significant benefits over full-fine tuning.
    -  **Time and memory efficiency**: With a large percentage of the parameters being frozen, the training time and the GPU memory is saved. Saving is more when using stateful optimizers like Adam, Adadelta etc.
    - **Storage efficiency**: No need to store huge checkpoints for different downstream tasks. Checkpoint size is greatly reduced with reduction in trainable parameters.
    - **No additional inference latency**: (unlike adaptors) just add the learned matrix to the pre-trained one.
    - **Easy task-switching in deployment**: all we need to change is a handful of weights as compared to the full model.
- Results:
    -  With GPT-3 175B, the VRAM consumption during training is reduced from 1.2TB to 350GB, and the trained checkpoint size reduced from 350GB to 35MB!!!
    - LoRA achieves performances comparable to and sometimes even better than fine-tuning the full model.

Note: “Low intrinsic rank” is inspired by the idea of “low intrinsic dimensionality” that these over-parameterized pre-trained models are seen to reside on, and that’s also the explanation behind why fine-tuning only a part of the full model rather than full fine-tuning can yield good results.

# Prompt Tuning 

| What | Prompt Tuning |
| ----- | ------------ |
| Description | A technique for fine-tuning a large language model (LLM) at inference time by learning a set of continuous, trainable parameters that modify the LLM's hidden states in response to task-specific prompts.|
| When to use | When you have a large pre-trained LLM and want to fine-tune it for multiple different downstream tasks at inference time with minimal computational resources. | 
| Benefits | More efficient and flexible than traditional fine-tuning, as it does not require retraining the entire model and allows you to fine-tune the model for multiple different tasks without having to retrain the model each time. | 

# Adapters

| What | Prompt Tuning |
| ----- | ------------ |
| Description | Small neural network modules that are added to pre-trained language models (LLMs) to adapt the model to new downstream tasks.|
| When to use | When you need to fine-tune multiple downstream tasks on the same pre-trained model | 
| Benefits | Adapters are more efficient than traditional fine-tuning, as they only require fine-tuning a small number of parameters. They are also more flexible, as they can be quickly and easily plugged into different parts of the pre-trained model without requiring major modifications.| 

# Prefix Tuning

| What | Prefix Tuning |
| ----- | ------------ |
| Description | A fine-tuning technique for pre-trained language models (LLMs) that involves adding a small trainable prefix to the input of the model. The prefix modifies the representation learned by the LLM to better suit the downstream task.|
| When to use | When you want to fine-tune a pre-trained LLM for a specific downstream task and have limited computational resources.| 
| Benefits | Prefix tuning is more efficient than traditional fine-tuning, as it only requires fine-tuning a small number of parameters. It is also more flexible, as it can be used to modify the representation learned by the LLM for a particular task.| 

# LoRa (Low-Rank Adaptation)

| What | Prefix Tuning |
| ----- | ------------ |
| Description | A fine-tuning technique for pre-trained language models (LLMs) that modifies the attention mechanism of the model by introducing a low-rank matrix factorization. This allows the model to learn task-specific attention patterns without having to retrain the entire model.|
| When to use | When you want to fine-tune a pre-trained LLM for a specific downstream task that requires task-specific attention patterns. It is also useful when you have limited computational resources and want to reduce the number of trainable parameters in the model.| 
| Benefits | LoRa is more efficient than traditional fine-tuning, as it only requires fine-tuning a small number of parameters. It is also more flexible, as it can be used to learn task-specific attention patterns for a particular task.| 

# Which Technique Should I Choose?
- Choosing a PEFT method is all about aligning with your objectives.

<hr/>

![peft](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/9a23575d-0e30-4168-8154-7cf1a9d01f77)

<hr/>

# References

| Papers | Resource | 
| ------- | :-------|
| Adapters (Paper)| [🔗](https://arxiv.org/pdf/1902.00751.pdf)|
| LoRA  (Paper) | [🔗](https://arxiv.org/abs/2106.09685)|
| Prefix Tuning  (Paper) | [🔗](https://arxiv.org/abs/2101.00190)|
| Prompt Tuning  (Paper) | [🔗](https://arxiv.org/abs/2104.08691)|
| LoRA (Repo)| [🔗](https://github.com/microsoft/LoRA)|
| HuggingFace PEFT (Repo)| [🔗](https://github.com/huggingface/peft)|
| Finetuning LLMs Efficiently with Adapters | [🔗](https://magazine.sebastianraschka.com/p/finetuning-llms-with-adapters)|
