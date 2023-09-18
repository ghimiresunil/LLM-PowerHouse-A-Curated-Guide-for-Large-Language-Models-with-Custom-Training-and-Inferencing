# Overview
- This tutorial will teach you how to organize your deep learning project in a logical way.
- To build a robust machine learning model or deep learning model, it is essential to split your data correctly into `Train set`, `Validation Set`, and `Test Set`. This tutorial will teach you the best practices for doing so.

# Choosing Train, Train-dev, Dev and Test Sets
- **Guideline**: To ensure that your model performs well on future data, choose a dev and test set that is representative of the data you expect to encounter.
    - To build a robust model that performs well on unseen data, the `dev` and `test` sets should be randomly sampled from the same distribution as the training set
    - When the `training (train)` and `development (dev)` sets must have different distributions due to insufficient data, it is helpful to introduce a `train-dev set` that is sampled from the same distribution as the training set. This `train-dev set` can be used to monitor overfitting and select the best model hyperparameters.
- **Guideline**: The `dev` and `test` sets should be just big enough to represent accurately the performance of the model.
    - The size of the `dev` and `test` set should be big enough for the `dev` and `test` results to be representative of the performance of the model. If the `dev` set has 100 examples, the `dev` accuracy can vary a lot depending on the chosen `dev` set. For bigger datasets (>1M examples), the `dev` and `test` set can have around **10,000 examples** each for instance (only 1% of the total data).

    | Dataset size	| Dev set size	| Test set size | 
    | -------------- | -------------- | ------------ |
    | Less than 1 million examples | 10% of the total data | 10% of the total data | 
    | Over 1 million examples	| 1% of the total data	| 1% of the total data |

    - What are the trade-offs between using a large dev and test set and using a small dev and test set?
        - There are a few reasons why it is recommended to use a larger dev and test set size for small datasets and a smaller dev and test set size for large datasets.
        - One reason is that small datasets are more likely to be overfit. Overfitting occurs when a model learns the training data too well and is unable to generalize to new data. Using a larger dev and test set for small datasets helps to reduce overfitting by ensuring that the model is evaluated on a more representative sample of the data.
        - Another reason is that small datasets are more likely to be noisy. Noise in data can lead to inaccurate model performance. Using a larger dev and test set for small datasets helps to reduce the impact of noise on the model's performance.
        - For large datasets, there is less risk of overfitting and noise. This is because large datasets are more likely to be representative of the overall population and less likely to contain outliers. Therefore, it is possible to use a smaller dev and test set for large datasets without sacrificing accuracy.
        - In addition to the above reasons, there is also a practical consideration when choosing the size of the dev and test sets. Training and evaluating a model on a large dataset can be computationally expensive and time-consuming. By using a smaller dev and test set for large datasets, it is possible to reduce the computational cost and time required to train and evaluate the model.
        - Here is a table that summarizes the advantages and disadvantages of using different dev and test set sizes for small and large datasets:

            | Dev and test set size		| Advantages	| Disadvantages | 
            | -------------- | -------------- | ------------ |
            | Large for small datasets	 | Reduces overfitting and noise	 | More computationally expensive and time-consuming | 
            | Small for large datasets		| Less computationally expensive and time-consuming		| More risk of overfitting and noise |

# Objectives
These guidelines translate into best practices for code:

- the split between `train` / `dev` / `test` should always be the same across experiments
    -  otherwise, different models are not evaluated in the same conditions
    - we should have a reproducible script to create the `train` / `dev` / `test` split
- we need to `test` if the dev and `test` sets should come from the same distribution

# Dataset Structure
- The best and most secure way to split the data into these three sets is to have one directory for `train`, one for `dev` and one for `test`.
- For instance if you have a dataset of images, you could have a structure like this with `80% `in the training set, `10%` in the dev set and `10%` in the `test` set.
```python
  data/
      train/
          img_000.jpg
          ...
          img_799.jpg
      dev/
          img_800.jpg
          ...
          img_899.jpg
      test/
          img_900.jpg
          ...
          img_999.jpg
```

# Reproducibility is Important
- Datasets often come in one big set, which you must split into `train`, `dev`, and `test` sets before starting your project. Academic datasets often come with a `train`/`test` split, but you must create your own `train`/`dev` split if it does not exist.
- Reproducibility is essential in machine learning, as it allows you to recreate the same exact `train`, `dev`, and `test` sets from scratch, ensuring that your results are reliable and verifiable.
- The cleanest way to do it is to have a `build_dataset.py` file that will be called once at the start of the project and will create the split into `train`, `dev` and `test`. Optionally, calling `build_dataset.py` can also download the dataset. We need to make sure that any randomness involved in `build_dataset.py` uses a fixed seed so that every call to `python build_dataset.py` will result in the same output.
- To ensure reproducibility, avoid splitting data manually, as this can be difficult to repeat accurately.
- Code snippet to split train, dev, and test set
```python
import argparse
import random
import os

from PIL import Image
from tqdm import tqdm


SIZE = 64

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/SIGNS', help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/64x64_SIGNS', help="Where to write the new data")


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contained in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # Use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_signs')
    test_data_dir = os.path.join(args.data_dir, 'test_signs')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
```

# Implementation Details
- Let’s illustrate the good practices with a simple example. We have filenames of images that we want to split into `train`, `dev` and `test`. Here is a way to split the data into three sets: `80% train`, `10% dev` and `10% test`.
```python
filenames = ['img_000.jpg', 'img_001.jpg', ...]

split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1]
dev_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]
```
## Ensure That Train/dev/test Belong to the Same Distribution (if Possible)
- Often we have a big dataset and want to split it into `train`, `dev` and `test` set. In most cases, each split will have the same distribution as the others.
- What could go wrong?
    - Suppose that we’ve divided our dataset into 10 groups. Let’s assume that the first 100 images (img_000.jpg to img_099.jpg) have label 0, the next 100 have label 1, … and the last 100 images have label 9.  
    - In that case, the above code will make the `dev` set only have label 8, and the `test` set only label 9.
- We therefore need to ensure that the filenames are correctly shuffled before splitting the data.
```python
filenames = ['img_000.jpg', 'img_001.jpg', ...]
random.shuffle(filenames)  # randomly shuffles the ordering of filenames

split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1]
dev_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]
```
- This should give approximately the same distribution for `train`, `dev` and `test` sets. If necessary, it is also possible to split each class into `80%`/`10%`/`10%` so that the distribution is the same in each set.

# Make it reproducible
- We talked earlier about making the script reproducible. Here we need to make sure that the `train`/`dev`/`test` split stays the same across every run of `python build_dataset.py`.
- The code above doesn’t ensure reproducibility, since each time you run it you will have a different split.
- To make sure to have the same split each time this code is run, we need to fix the random seed before shuffling the filenames
- Here is a good way to remove any randomness in the process:

```python
filenames = ['img_000.jpg', 'img_001.jpg', ...]
filenames.sort()  # make sure that the filenames have a fixed order before shuffling
random.seed(230)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))
train_filenames = filenames[:split_1]
dev_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]
```
- The call to `filenames.sort()` makes sure that if you build filenames in a different way, the output is still the same.

# References
- [Structuring Machine Learning Projects on Coursera](https://www.coursera.org/learn/machine-learning-projects)
- [CS230 code examples](https://github.com/cs230-stanford/cs230-code-examples/tree/master)
