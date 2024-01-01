# Sharding 
Before we dive into how to shrink our model's appetite for VRAM, let's talk about another cool trick: chopping it up! We can split our model into smaller bite-sized pieces called "shards," which frees up memory space.

Think of each shard as a mini-model. Instead of overloading one GPU, we spread the model's weight across several GPUs, giving them smaller pieces to handle. This keeps everyone happy and avoids memory meltdowns.

The model that loaded below, [Zephyr-7B-β](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), was actually already sharded for us! If you go to the model and click the “Files and versions” link, you will see that the model was split up into eight pieces.

Think of slicing the model into shards like chopping vegetables for a big cookout. But before you grab the knife, check if pre-chopped "quantized" ingredients are available! If not, mastering the art of "quantization" lets you chop them yourself for smaller, tastier (more efficient) results.

```python
# Shard our model into pieces of 1GB
accelerator = Accelerator()
accelerator.save_model(
    model=pipe.model, 
    save_directory="/content/model", 
    max_shard_size="4GB"
)
```

And that is it! Because we sharded the model into pieces of 4GB instead of 2GB, we created fewer files to load:


