from torch.utils.data import DataLoader
from torch import cuda
import math

from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

from datasets import load_dataset
from datasets import DatasetDict

import os
import logging
import argparse


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### print debug information to stdout

logging.info(f"CUDA Device Name:{cuda.get_device_name()}")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--dataset_path",
        type= str,
        required= True,
        help="Enter the file path for training sentence transformers model in csv, json, huggingface datset"
    )
    
    parser.add_argument(
        "--model_name",
        type= str,
        required= True,
        help="Model name that you want to train embeddings models on."
    )
    
    parser.add_argument(
        "--output_dir",
        type= str,
        required=True,
        help="Enter the output directory you want your model to be saved"
    )
    
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Enter the batch size you want data to be during training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of epochs to train model on"
    )
    
    return parser.parse_args()

def split_training_data(data, test_size, valid_size):
    train_test = data["train"].train_test_split(test_size=test_size)
        
    test_valid = train_test["test"].train_test_split(test_size=valid_size)
    
    return DatasetDict({
        'train': train_test['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']})

def train(dataset_path, model_name, output_dir, train_batch_size, num_epochs):
    
    file_extension = dataset_path.split(".")
    
    if file_extension == "csv":
        dataset = load_dataset(
            "csv",
            data_files = dataset_path
        )
            
    elif file_extension == "json":
        dataset = load_dataset(
            "json",
            data_files = dataset_path
        )
        
    else:
        dataset = load_dataset(
            dataset_path
        )
        
    if set(["validation","test"]).issubset(set(dataset.keys())):
        train_test_valid_dataset = dataset
    
    else:
        train_test_valid_dataset = split_training_data(data= dataset)
        
    model_save_path = os.path.join(output_dir, model_name)
    
    word_embedding_model = models.Transformer(model_name)
    
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Convert the dataset to a DataLoader ready for training
    logging.info("Read stsb-multi-mt train dataset")

    train_samples = []
    dev_samples = []
    test_samples = []

    def samples_from_dataset(dataset):
        samples = [InputExample(texts=[e['sentence1'], e['sentence2']], label=e['similarity_score'] / 5) \
            for e in dataset]
        return samples
    
    train_samples = samples_from_dataset(train_test_valid_dataset["train"])
    dev_samples = samples_from_dataset(train_test_valid_dataset["valid"])
    test_samples = samples_from_dataset(train_test_valid_dataset["test"])

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    initial_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, write_csv=False)
    initial_evaluator(model)

    logging.info("Read stsb-multi-mt dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    ## Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=num_epochs,
            evaluation_steps=1000,
            warmup_steps=warmup_steps,
            output_path=model_save_path)
    
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='stsb-multi-mt-test')
    test_evaluator(model, output_path=model_save_path)

if __name__=="__main__":
    args = parse_args()
    
    train(args.dataset_path, args.model_name, args.output_dir, args.train_batch_size, args.num_epochs)
    
    