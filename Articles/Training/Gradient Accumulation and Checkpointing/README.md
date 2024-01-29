# Gradient Accumulation and Gradient Checkpointing

Big models, big problems? Don't let a lack of GPU memory hold you back! This article tackles a common struggle in deep learning: training complex models without getting the dreaded "CUDA: out of memory" error. We'll explore smart tricks to squeeze more efficiency out of your GPU memory and boost your training speed, all without getting tangled in technical jargon. Get ready to unlock the secrets of memory optimization and train those giants like a pro!

Are you struggling with out-of-memory issues when training your Deep Learning models? Gradient checkpointing can help you free up some memory. Today, let's dive into Gradient Checkpointing.

## Gradient Accumulation
- Imagine learning from a pile of books. Usually, you'd read one book, update your understanding, and then move on. But if the pile is HUGE, carrying all the books at once is tough! Gradient accumulation is like reading one chapter from each book before updating your brain (the model). Instead of constantly carrying all the book summaries (gradients), you save them up until you have a manageable chunk, then update your brain with the combined knowledge. This saves memory and lets you learn from even the biggest piles of data!
- For Example:
    - Instead of learning after each small chapter (batch) of a book, imagine reading four chapters in a row and taking notes on them all together. Then, after those four chapters, you use all your collected notes to update your overall understanding of the book (the model). This way, you don't have to remember everything after each chapter, saving your brain space (memory) for bigger things!
    - This "four-chapter note-taking" is like gradient accumulation with four batches. You combine the updates from multiple batches, saving memory by updating less often, and allowing you to tackle bigger books (complex models) or read in bigger chunks (larger batches).
    - This avoids technical jargon and keeps the analogy of learning from books to make it easily understandable. It also explains the benefits of accumulating gradients in terms of memory and handling larger tasks.
- Gradient accumulation is a technique used in deep learning to increase the effective batch size during training. Normally, the weights of a neural network are updated based on the gradients computed from a single batch of training data. However, for larger models or datasets, the batch size may be limited by the memory capacity of the GPU, leading to a significantly longer time to convergence due to vectorization.
- For efficient training, gradient accumulation breaks down data batches into smaller chunks, feeding them sequentially to the neural network and cumulatively summing the resulting gradients before adjusting the network parameters.
- After accumulating sufficient gradients through the aforementioned process, we perform the model's optimization step (using the standard `optimizer.step()`) to update the model parameters effectively.
- The following code illustrates how gradient accumulation improves the model's performance.

```python
training_args = TrainingArguments(per_device_train_batch_size=1, gradient_accumulation_steps=4, **default_args)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)

#result 
> BEFORE
Time: 57.82
Samples/second: 8.86
GPU memory: 14949 MB

> AFTER
Time: 66.03
Samples/second: 7.75
GPU memory: 8681 MB
```
- While gradient accumulation offers the advantage of larger effective batch sizes (particularly beneficial for contrastive learning) in memory-constrained settings, it can also lead to slower convergence and longer training times due to delayed parameter updates.
- Below code demonstrates the core principle of gradient accumulation. It trains a loop with `num_iterations`, where weights are updated every `accumulation_step` mini-batches.
- Each iteration in this code accumulates gradients from accumulation_steps mini-batches via `compute_gradients()` before updating weights with `update_weights()`.
```python
# Training loop

for i in range(num_iterations):
    accumulated_gradients = 0
    for j in range(accumulation_steps):
        batch = next(training_batch)
        gradients = compute_gradients(batch)
        accumulated_gradients += gradients
    update_weights(accumulated_gradients)
```

## Gradient Checkpointing
- To balance memory usage and computation time in backpropagation, especially for deep neural networks with many layers or memory constraints, gradient checkpointing strategically recomputes intermediate activations during the backward pass, rather than storing them all upfront.
- Gradient checkpointing mitigates memory demands by selectively recomputing essential intermediate activations during backpropagation, caching only those crucial for gradient computation, and thereby trading reduced memory usage for increased computation time.
- Through the strategic selection of which intermediate activations to checkpoint, based on memory constraints and computational trade-offs, gradient checkpointing promotes memory-efficient training, unlocking the potential for larger model architectures and alleviating memory bottlenecks in deep learning endeavors.
- Gradient checkpointing strategically conserves memory during backpropagation, particularly in deep neural networks brimming with layers or parameters, by selectively recomputing intermediate activations rather than storing them all.
- To circumvent the memory-intensive process of storing all intermediate activations during backpropagation, gradient checkpointing strategically stores only a select subset, recomputing the remainder on-demand during the backward pass, thereby significantly reducing memory requirements during training.
- While entirely forgoing activation storage during the forward pass and recomputing them all during backpropagation would indeed lessen memory demands, it would incur a considerable computational cost, potentially hindering training efficiency.
- This strategic memory exchange opens doors for training larger models or leveraging bigger batch sizes, both previously limited by memory constraints.
- The code below ([source](https://huggingface.co/docs/transformers/v4.18.0/en/performance)), with addition of gradient checkpointing along with gradient accumulation, we can see that some memory is saved but the training time has become slower. As [HuggingFace](https://huggingface.co/docs/transformers/v4.18.0/en/performance) mentions, a good rule of thumb is that gradient checkpointing slows down training by 20%.
```python 
training_args = TrainingArguments(
    per_device_train_batch_size=1, gradient_accumulation_steps=4, gradient_checkpointing=True, **default_args
)

trainer = Trainer(model=model, args=training_args, train_dataset=ds)
result = trainer.train()
print_summary(result)

# output

> BEFORE
Time: 66.03
Samples/second: 7.75
GPU memory: 8681 MB

> AFTER
Time: 85.47
Samples/second: 5.99
GPU memory occupied: 6775 MB.
```

## Summary
-  To address memory limitations in deep learning training, gradient accumulation strategically accumulates gradients over multiple batches before updating model parameters, while gradient checkpointing efficiently recomputes select intermediate activations, reducing memory usage during backpropagation. Both techniques offer valuable optimization strategies for memory and computational resources.

## Reference

| Source | Link |
| ------- | :-----: |
| Raz Rotenberg’s What is Gradient Accumulation in Deep Learning? | [🔗](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa) |
| Hugging Face’s Perfomance and Scalability | [🔗](https://huggingface.co/docs/transformers/v4.18.0/en/performance) |
| Yaroslav Bulatov’s Fitting larger network into memory | [🔗](https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9) |
