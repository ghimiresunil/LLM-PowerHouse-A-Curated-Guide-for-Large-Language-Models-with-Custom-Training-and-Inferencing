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

