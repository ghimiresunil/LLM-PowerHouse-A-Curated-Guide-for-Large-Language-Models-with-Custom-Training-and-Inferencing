# Overview
- Have you ever wondered why, at times, the training loss is higher than the validation loss?

# Theories
- Here are several hypotheses that could explain this phenomenon.
![train_loss_is_greater_than_val_loss](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/4940a943-24a6-407b-99d5-e30ae8b5bfd0)

- **Regularization**: Regularization, such as dropout, is often the cause, as it is applied during training but not during validation and testing. When the regularization loss is added to the validation loss, the results change.
![regularization](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/9b5bcaa5-fd95-4aa9-9005-7fdf198e007d)

    - For Example:
        - Suppose that your model's training loss without regularization is 0.5. When you add the dropout regularization loss, the training loss increases to 0.6.
        - However, the validation loss remains at 0.5, because the dropout regularization loss is not applied during validation.
        - This means that the training loss is greater than the validation loss.
        - If you add the dropout regularization loss to the validation loss, the validation loss also increases to 0.6. This makes the training and validation losses equal.
        - This example shows how regularization can cause training loss to be greater than validation loss. However, it is important to note that this is a good thing. Regularization helps to prevent overfitting, which can lead to better performance on unseen data.
- **Epoch delta between training and validation loss**: Training loss is measured during each epoch, while validation loss is measured after each epoch. This means that training loss is measured half an epoch earlier, on average. If we shift the training loss curve half an epoch to the left, the two curves look much more similar.
![Epoch delta between training and validation loss](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/0cc0a92d-ae50-4162-a28c-7db91bdb8170)

    - For Example: 
        - Imagine you have a training set of 100 images, and you are training a model to classify the images into two categories: cats and dogs. You split the training set into two batches of 50 images each.
        - You train your model on the first batch of images, and then you measure the training loss. Next, you train your model on the second batch of images, and then you measure the training loss again.
        - Finally, you evaluate your model on the entire training set, and you measure the validation loss.
        - In this example, the training loss is measured half an epoch earlier than the validation loss. This is because the training loss is measured after each batch of data is passed through the model, while the validation loss is measured after all of the training data has been passed through the model.
        - If we shift the training loss curve back half an epoch, the two curves look much more alike:
        - Before shift:
            ```python
            Training loss: [1.0, 0.9, 0.8, 0.7, 0.6]
            Validation loss: [0.9, 0.8, 0.7, 0.6, 0.5]
            ```
        - After shift:
            ```python
            Training loss: [0.9, 0.8, 0.7, 0.6, 0.5]
            Validation loss: [0.9, 0.8, 0.7, 0.6, 0.5]
            ```
- **Easier validation set**: It is possible that the validation set is easier than the training set. This can happen by chance if the validation set is too small, or if it is not sampled properly (e.g., too many easy classes).
    - For Example: 
        - Imagine you are training a model to classify images of cats and dogs. You have a large training set of images, but you only have a small validation set. You split the training set into two batches, and you use one batch to train the model and the other batch to validate the model.
        - By chance, the validation set happens to contain only images of cats. This means that the validation set is much easier than the training set, because the model does not have to learn to distinguish between cats and dogs.
- **Data leaks**: It might also be possible that the training set leaked into the validation set.
    - For Example:
        - Imagine you are training a model to predict the price of houses. You split the training set into two batches, and you use one batch to train the model and the other batch to validate the model.
        - However, you accidentally preprocess the validation set in the same way as the training set. This means that the model has already learned some of the patterns in the validation set, and it will perform better on the validation set than it would on unseen data.
- **Data augmentation**: Data augmentation during training can also lead to training loss being greater than validation loss.
    - For Example:
        - If we randomly crop training images, and 10% of the time the crop doesn't include the main object, the training images will be harder to classify than the validation images.
        - Another common case is when the data augmentation algorithm uses many transformations to create training images that are more diverse (in lighting, rotation, scale, etc.) than the validation images. This can make the training images more difficult to classify than the validation images.
    - To validate the theory that training loss is greater than validation loss because the training images are more difficult to classify, we can compare the training and validation losses using the same augmentation procedure for both training and validation sets.
        - **Do not use** the same augmentation procedure for validation if you are using early stopping or comparing models, because in these cases, you are only interested in the performance on the test set, not the training set.
> Note: A close validation loss to training loss does not guarantee that your model is not overfitting.

# Remedies
- To compare training and validation losses on an equal footing, add the regularization loss to the training loss.
- Shift the training loss by half an epoch.
- Make sure the validation set is large enough.
- To avoid data leakage, the validation set should be chosen from the same distribution as the training set, without using any information from the training set to select the samples.

# References
- [Aurélien Geron’s Twitter](https://twitter.com/aureliengeron/status/1110839223878184960) for the great inputs.







