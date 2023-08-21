import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def train_doc2vec_model(clean_resume_dir, model_save_path):
    """
    Trains a Doc2Vec model on tokenized resumes and saves the model to the specified path.
    
    :param clean_resume_dir: Path to the directory containing clean resume files.
    :param model_save_path: Path to save the trained model.
    """
    tokenized_resumes = []

    # Tokenize and preprocess resumes
    for clean_resume in os.listdir(clean_resume_dir):
        with open(os.path.join(clean_resume_dir, clean_resume), "r") as file:
            resume = file.read()
            tokens = word_tokenize(resume)
            tokenized_resumes.append(tokens)

    # Prepare TaggedDocument format for training
    tagged_data = [TaggedDocument(words=d, tags=[i]) for i, d in enumerate(tokenized_resumes)]

    # Define and train the Doc2Vec model
    model = Doc2Vec(
        vector_size=20,     # Dimensionality of the feature vectors
        window=2,            # Maximum distance between the current and predicted word within a sentence
        min_count=10,        # Ignores all words with total frequency lower than this
        dm=1,                # Distributed Memory (PV-DM) architecture
        dm_mean=1,           # Use mean of context vectors for PV-DM
        epochs=100,          # Number of iterations (epochs) over the corpus
        seed=42,             # Seed for reproducibility
        workers=6            # Number of worker threads for training
    )

    # Build vocabulary and train the model
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Save the trained model
    model.save(model_save_path)

# Specify the paths and call the function
clean_resume_dir = "../data/clean_resume/"
model_save_path = "skill_doc2vec.model"
train_doc2vec_model(clean_resume_dir, model_save_path)
