# Overview
- In this article, we will begin the process of categorizing and delving deeply into various types of word embeddings utilized in NLP, along with their respective use cases.

# What is Natural Language Processing?

<img align="right" width="400" src="https://user-images.githubusercontent.com/40186859/220372533-59c81502-85ac-49d5-947a-13341d080f30.png" />

Natural language processing is a multidisciplinary field that combines techniques from computer science,  linguistics, and artificial intelligence to develop algorithms and models that enable machines to understand, analyze, and generate human language, such as text, speech, and even gestures.

For example:
- One application of natural language processing is virtual assistants like Apple's Siri or Amazon's Alexa, which can understand spoken commands and respond to them in natural language. 
- Another example is sentiment analysis, which uses machine learning to analyze social media posts and identify the sentiment or emotion behind them. This technology can help companies understand how people feel about their products or services, and improve their marketing strategies accordingly.
- Here's a funny example of NLP

Q: Why did the computer go to the doctor?

A: Because it had a virus! The doctor used natural language processing to diagnose the issue and prescribed some anti-virus software to help the computer get better.

Of course, this is just a joke, but it highlights the idea that natural language processing can be used to analyze and understand human language, even when it's used in a humorous context. In reality, NLP has many practical applications in fields like healthcare, finance, and customer service, but it's always fun to imagine what kind of conversations machines could have if they were truly fluent in natural language.

# Working with Text Data

Text data poses unique challenges and requires distinct solutions compared to other types of datasets. Preprocessing and cleaning of text data is often more intensive than with other data formats, in order to prepare it for statistical analysis or machine learning.

# What is vectorization?

- Vectorization" is a term used in the context of converting input data, such as text, into numerical vectors to make it compatible with machine learning models. This approach has been in use since the advent of computers and has proven to be highly effective across various fields, including natural language processing
- In Machine Learning, vectorization is a step in feature extraction. The idea is to get some distinct features out of the text for the model to train on, by converting text to numerical vectors.

Points to be remember:

Most simple of all the simple techniques involves three operation

- Tokenization: First, the input text is tokenized. A sentence is represented as a list of its constituent words, and it’s done for all the input sentences.

- Vocabulary creation: Of all the obtained tokenized words, only unique words are selected to create the vocabulary and then sorted by alphabetical order.

- Vector Creation: Finally, a sparse matrix is created for the input, out of the frequency of vocabulary words. In this sparse matrix, each row is a sentence vector whose length (the columns of the matrix) is equal to the size of the vocabulary.

## Bag of Words

<img align="right" width="400" src="https://user-images.githubusercontent.com/40186859/220844654-f3f931e5-a03f-4bfe-a5cd-55211510a505.png" />

The most common approach to working with text involves vectorizing it by creating a Bag of Words, which accurately describes the final product as containing information about all important words in the text individually, but not in any particular order. This process involves throwing every word in a corpus into a bag, which, with a large enough corpus, reveals certain patterns that may emerge. For example, a bag of words made from Shakespeare's Hamlet is likely more similar to a bag of words made from Macbeth than to something like The Hunger Games. The simplest way to create a bag of words is to count how many times each unique word is used in a corpus, and having a number for every word enables us to treat each bag as a vector, thereby opening up all kinds of machine learning tools for use

Now let's explore bag of word in more details.

The CountVectorizer from Scikit-learn library is commonly used for creating a bag of words representation of text. The CountVectorizer converts a collection of text documents into a matrix of token counts, where each row represents a document, each column represents a word, and the values in the matrix are the frequency counts of each word in the corresponding document. This approach to text vectorization is useful for various machine learning tasks such as classification, clustering, and information retrieval.

It's natural to feel curious about how sentiment analysis works. When it comes to analyzing textual data for sentiment, we can't simply fit the raw text into a model. First, we need to convert the text into a numerical format using vectors, which is a common approach in NLP.

Suppose I have some positive examples over here:

**Sentence 01**: He is an intelligent boy

**Sentence 02**: She is an intelligent girl

**Sentence 03**: Both boy and girl are an intelligent 

During text preprocessing, there are a few tasks we need to perform. First, we need to convert the sentence to lowercase. Next, we should remove any stop words, which are commonly used words that don't carry much meaning, such as "the", "and", or "is". Finally, we can apply stemming or lemmatization, which involves reducing words to their root form, in order to further standardize the text.

After applying these steps, the resulting sentence would be transformed into a cleaner and more uniform representation that is more suitable for analysis or modeling.

**Sentence 01**: intelligent boy

**Sentence 02**: intelligent girl

**Sentence 03**: boy girl intelligent 

Text processing is the initial step, but our main focus is on how to derive vectors using bag of words. 🤔

To achieve this, we need to analyze each word in the pre-processed sentence and determine its frequency. By doing so, we can create a representation of the text in the form of a vector

| Words | Frequency |
|------ | :----------: |
| Intelligent | 3 |
| boy | 2 |
| girl | 2 |

Note: When calculating word frequency, the words may not be in order, but it's essential to sort them in descending order to make it easier to analyze the most important words in the text.

Now let's appy `Bag of Words`

|  | $f_1$| $f_2$ | $f_3$ |
|------ |------ | ---------- | ----------|
|  | intelligent | boy | girl|
| vector of sentence 01 | 1 | 1 | 0 |
| vector of sentence 02 | 1 | 0 | 1 |
| vector of sentence 03 | 1 | 1 | 1 |

Finally we have derived vectors using bag of words.😊 While bag of words is a useful technique for text analysis, it also has its disadvantages.  One of the main disadvantages is that it doesn't take into account the context in which words appear. As a result, words with multiple meanings can be assigned the same vector representation, leading to ambiguity in the analysis. Additionally, bag of words doesn't capture the relationship between words, such as synonyms and antonyms, which can affect the accuracy of natural language processing tasks.

<b> Listed drawback of using a bag of words </b>

- If the new sentences contain new words, then our vocabulary size would increase and thereby, the length of vector would increase too.
- Additionally, the vectors would also contain many 0s, thereby resulting in a sparse matrix (which is what we would like to avoid)
- We are retaining no information on the grammar of the sentences nor the ordering of the words in the text.

From the above-generated vectors, it can be observed that the values assigned to the vectors are either 1 or 0. However, an important point to note is that both "intelligent" and "boy" have been assigned a value of 1, despite having different semantic meanings. This makes it difficult to determine which word holds greater importance. In sentiment analysis, it is crucial to identify the words that carry more weightage in determining the sentiment.

To overcome some of the limitations of the bag-of-words model we have something call as `TF-IDF` which is also called `Term Frequency and inverse document frequency` 😊 

## TF-IDF
Term Frequency - Inverse Document Frequency (TF-IDF) is a numerical statistics that is intended to reflect how important a word is to a document in a collection or corpus. 

Term Frequency (TF) - If is a measure of how frequently a term, t appears in a document d. 

$$t_{f_{t_i}d} = \frac{n_{t_id}}{Number\ of\ terms\ in\ the\ document}$$

Here, in the numerator, n is the number of times the term 't' appears in the document 'd'. Thus, each document and term would have its own TF value.

Let's take an example:

- This movie is very scary and long
- This movie is not scary and is slow
- This movie is spooky and good

First we will build a vocabulary from all unique words in above three movie reviews. The vocabulary consists of these 11 words.

Vocabulary: 'This', 'movie, 'is', 'very', 'scary', 'and', 'long', 'not, 'slow', 'spooky', 'good' 

- Number of words in first example: 7
- Number of words in second example: 8
- Number of word in third example: 6 

Example:

TF of the word `this` in second sentence: = $\frac{number\ of\ times\ this\ appear\ in\ second\ sentence}{number\ of\ terms\ in\ second\ sentence}$ = $\frac{1}{8}$

We can calculate the term frequencies for all terms and all the sentence in the number

| Term | Sentence 1 | Sentence 2 | Sentence 3 | TF Sentence 1 | TF Sentence 2 | TF Sentence 3|
| ---- | ---| ------- | -------  |-------  | -------  | ------- |
| This | 1 | 1 | 1 | $\frac{1}{7}$ | $\frac{1}{8}$ | $\frac{1}{6}$ |
| movie | 1 | 1 | 1 | $\frac{1}{7}$ | $\frac{1}{8}$ | $\frac{1}{6}$ |
| is | 1 | 2 | 1 | $\frac{1}{7}$ | $\frac{1}{4}$ | $\frac{1}{6}$ |
| very | 1 | 0 | 0 | $\frac{1}{7}$ | 0 | 0 |
| scary | 1 | 1 | 0 | $\frac{1}{7}$ | $\frac{1}{8}$ | 0 |
| and | 1 | 1 | 1 | $\frac{1}{7}$ | $\frac{1}{8}$ | $\frac{1}{6}$ |
| long | 1 | 0 | 0 | $\frac{1}{7}$ | 0 | 0 |
| not | 0 | 1 | 0 | 0 | $\frac{1}{8}$ | 0|
| slow | 0 | 1 | 0 | 0 | $\frac{1}{8}$ | 0|
| spooky | 0 | 0 | 1 | 0 | 0 | $\frac{1}{6}$|
| good | 0 | 0 | 1 | 0 | 0 | $\frac{1}{6}$|

Inverse Document Frequency (IDF): IDF is the measure of how important a term is. We need the IDF value because TF alone is not sufficient to understand the importance of words.

$idf_i$ = $log\frac{Number\ of\ Documents}{Number\ of\ documents\ with\ term\ 't'}$

Example:

Let's calculate the IDF value of word `this` in sentence 2.

IDF `this` in sentence 2 = $log\frac{Number\ of\ documents}{Number\ of\ documents\ containing\ the\ word\ this}$ = $log\frac{3}{3}$ = log(1) = 0

The IDF Values for the entire vocabulary would be:

| Term | Sentence 1 | Sentence 2 | Sentence 3 | IDF |
| ------ | ------ | -------- | ------ | ------ |
| This | 1 | 1 | 1 | $log\frac{3}{3}$ = 0 | 
| movie | 1 | 1 | 1 | $log\frac{3}{3}$ = 0 |
| is | 1 | 2 | 1 | $log\frac{3}{3}$, $log\frac{3}{3}$  = 0 |
| very | 1 | 0 | 0 | $log\frac{3}{1}$ = 0.48 |
| scary | 1 | 1 | 0 | $log\frac{3}{2}$ = 0.18 |
| and | 1 | 1 | 1 | $log\frac{3}{3}$ = 0|
| long | 1 | 0 | 0 | $log\frac{3}{1}$ = 0.48|
| not | 0 | 1 | 0 | $log\frac{3}{1}$ = 0.48|
| slow | 0 | 1 | 0 | $log\frac{3}{1}$ = 0.48|
| spooky | 0 | 0 | 1 | $log\frac{3}{1}$ = 0.48|
| good | 0 | 0 | 1 | $log\frac{3}{1}$ = 0.48|


We can observe that certain words such as "is", "the", and "and" have been assigned a value of 0, indicating their lower significance. In contrast, words such as "scary", "long", and "good" have a higher value, indicating their importance. By calculating the TF-IDF score for each word in the corpus, we can determine their respective importance levels.

$TF-IDF_{t,d}$ = $TF_{t,d}$  * $IDF_t$

| Term | Sentence 1 | Sentence 2 | Sentence 3 | IDF | TF-IDF Sentence 01 | TF-IDF Sentence 02 | TF-IDF Sentence 03 | 
| ------ | ------ | -------- | ------ | ------ | ------ | ------ | ------ | 
| This | 1 | 1 | 1 |  0 | 0 | 0 | 0 |
| movie | 1 | 1 | 1 |  0 | 0 | 0 | 0 |
| is | 1 | 2 | 1 | 0 | 0 | 0 | 0 | 
| very | 1 | 0 | 0 | 0.48 | 0.068 | 0 | 0 | 
| scary | 1 | 1 | 0 | 0.18 | 0.025 | 0.022 | 0 |
| and | 1 | 1 | 1 | 0| 0 | 0 | 0 |
| long | 1 | 0 | 0 | 0.48| 0.068 | 0 | 0 
| not | 0 | 1 | 0 | 0.48| 0 | 0.060 | 0 |
| slow | 0 | 1 | 0 | 0.48| 0 | 0.060 | 0 |
| spooky | 0 | 0 | 1 | 0.48| 0 | 0 | 0.080|
| good | 0 | 0 | 1 | 0.48| 0 | 0 | 0.80 | 

After calculating the TF-IDF scores for our vocabulary, it became evident that less frequent words were given higher values, indicating their relative importance in the corpus. TF-IDF scores were found to be particularly high for words that were rare in all documents combined, but frequent in a single document, indicating their potential significance in that particular context.

### Problem of Bag of Worrds and TF-IDF
- Both BOW and TF-IDF approach semantic information is not stored. TF-IDF gives importance to uncommon words
- There is definately chance of overfitting 

To overcome such problem of BOW and TF-IDF we use technique called Word2Vec.

## Word2vec

Word2Vec is a technique for natural language processing published in 2013 by Google. Word2Vec is a word embedding technique that learns to represent words as vectors in a high-dimensional space. These vectors are designed to capture the semantic meaning of words, so that words that are similar in meaning will have similar vectors.

There are two main types of Word2Vec models: CBOW (Continuous Bag of Words) and Skip-Gram. 

In CBOW, the model is trained to predict a word given its surrounding words. For example, the model might be trained to predict the word "dog" given the words "the", "cat", and "chased". 

In Skip-Gram, the model is trained to predict the surrounding words given a word. For example, the model might be trained to predict the words "the", "cat", and "chased" given the word "dog".

The Word2Vec model learns to represent words as vectors by using a neural network. The neural network has one input layer for each word in the vocabulary, and one output layer for each word in the vocabulary. The weights of the neural network are adjusted so that the model can predict the surrounding words of a given word with high accuracy.

Once the Word2Vec model is trained, it can be used to do a variety of tasks, such as:
- Finding similar words: Given a word, the model can be used to find words that are similar in meaning. For example, given the word "dog", the model might return the words "cat", "animal", and "pet".
- Analyzing text: The model can be used to analyze text for patterns and trends. For example, the model could be used to find out which words are most commonly used together, or to identify the topic of a document.
- Generating text: The model can be used to generate new text that is similar to the text it was trained on. For example, the model could be used to write a poem or a short story.

Here is an example of how Word2Vec can be used to find similar words. The following sentence is given to the Word2Vec model:

Example: The cat chased the dog.

The model is able to predict the surrounding words of each word in the sentence, with high accuracy. For example, the model is able to predict that the word "cat" is likely to be followed by the word "chased". The model can then use this information to find words that are similar to "cat". In this case, the model might return the words "dog", "animal", and "pet".

Word2Vec is a powerful tool that can be used to understand and analyze text. It is used in a variety of applications, such as machine translation, question answering, and sentiment analysis.

### Difference bwetween BOW, TF-IDF and Word2vec
Let's discuss the differences between Bag of Words (BOW), Term Frequency-Inverse Document Frequency (TF-IDF), and Word2Vec. In BOW, we obtain a sparse matrix with either 0 or 1 values, while in TF-IDF, we may get decimal values ranging from 0 to 1. However, Word2Vec works differently. To illustrate this, let's consider a vocabulary comprising the unique words in a given corpus.

For example, let's say our vocabulary consists of the words BOY, GIRL, KING, QUEEN, APPLE, and MANGO. And let's say our corpus consists of the following two sentences:

- The boy ate the apple.
- The queen wore a mango dress.

In BOW, we would create a sparse matrix with six rows (one for each word in the vocabulary) and two columns (one for each sentence). The values in the matrix would be either 0 or 1, indicating whether the word appears in the sentence or not. For example, the matrix would look like this:

 Word | Sentence 1 | Sentence 2 |
|---|---|---|
| BOY | 1 | 0 |
| GIRL | 0 | 0 |
| KING | 0 | 0 |
| QUEEN | 0 | 1 |
| APPLE | 1 | 0 |
| MANGO | 0 | 1 |

In TF-IDF, we would calculate the term frequency (TF) and inverse document frequency (IDF) for each word in each sentence. The TF for a word is the number of times it appears in a sentence, divided by the total number of words in the sentence. The IDF for a word is the logarithm of the number of documents in the corpus, divided by the number of documents that contain the word. For example, the TF-IDF matrix for our corpus would look like this:

| Word | Sentence 1 | Sentence 2 |
|---|---|---|
| BOY | 0.5 | 0 |
| GIRL | 0 | 0 |
| KING | 0 | 0 |
| QUEEN | 0 | 0.5 |
| APPLE | 0.5 | 0 |
| MANGO | 0 | 0.5 |

As you can see, the TF-IDF values are not binary like they are in BOW. Instead, they are decimal values that indicate the importance of a word in a sentence.

Word2Vec works differently than BOW and TF-IDF. It creates a vector representation for each word in the vocabulary. These vectors are real numbers, and they are trained on a corpus of text. The vectors for words that are semantically similar will be close together in vector space. For example, the vectors for the words BOY and GIRL would be close together, as would the vectors for the words KING and QUEEN.

In above example, the vectors for the words BOY and GIRL would be close together because they are both nouns that refer to people. Similarly, the vectors for the words KING and QUEEN would be close together because they are both nouns that refer to royalty.

You can also see this by looking at the cosine similarity between the vectors for different words. The cosine similarity is a measure of how similar two vectors are. In our example, the cosine similarity between the vectors for BOY and GIRL would be very high, and the cosine similarity between the vectors for KING and QUEEN would also be very high.

Word2Vec is a powerful tool for natural language processing because it can be used to understand the semantic relationships between words. This can be used for a variety of tasks, such as text classification, sentiment analysis, and question answering.

## GloVe (Global Vectors for Word Representation)

GloVe (Global Vectors for Word Representation) is an unsupervised learning algorithm for obtaining vector representations for words. It was proposed by Jeffrey Pennington, Richard Socher, and Christopher Manning in 2014.

GloVe works by training on aggregated global word-word co-occurrence statistics from a corpus. This means that it does not consider the context of a word in a sentence, but rather the overall frequency of words appearing together. This makes GloVe faster to train than some other word embedding models, such as Word2vec, which considers the context of a word.

The resulting word vectors from GloVe have been shown to be effective for a variety of natural language processing tasks, including word similarity, named entity recognition, and machine translation.

Here are some of the key features of GloVe:
- It is an unsupervised learning algorithm, which means that it does not require any labeled data. This makes it more scalable than supervised learning algorithms, which can be difficult to train on large datasets.
-  It is fast to train, even on large datasets. This is because it does not consider the context of a word in a sentence, but rather the overall frequency of words appearing together.
- The resulting word vectors are effective for a variety of natural language processing tasks.

## FastText

FastText is an open-source library for learning of word embeddings and text classification created by Facebook's AI Research (FAIR) lab. It is designed to efficiently handle large amounts of text data and provides tools for text classification, word representation, and text similarity computation.

FastText builds on the word2vec model, but it has several key differences. First, FastText uses a hierarchical classifier to train the model, which is faster and more efficient than word2vec. Second, FastText can provide embeddings for out-of-vocabulary (OOV) words, while word2vec cannot. This is because FastText also learns embeddings for character n-grams, which can be used to represent OOV words. Third, FastText has been shown to be more effective than word2vec for text classification tasks.

Here are some of the key features of FastText:
- It is fast and efficient, even for large datasets.
- It can provide embeddings for OOV words.
- It is effective for text classification tasks.
- It is open-source and free to use.

Here is how FastText works:
- The first step is to preprocess the text data. This involves tokenizing the text and removing stop words.
- The second step is to train the model. FastText uses a hierarchical classifier to train the model. The classifier is trained on a large corpus of text data.
- The third step is to use the model to generate word embeddings. Word embeddings are vector representations of words. They are used to represent words in a way that is meaningful to machines.
- The fourth step is to use the word embeddings for downstream tasks. For example, the word embeddings can be used for text classification, spam filtering, or question answering.

# Contextual Embeddings (considers Order and Context of Words)
| Embedding | ELMo (Embeddings from Language Models) | BERT (Bidirectional Encoder Representations from Transformers)|
| ---------- | ----- |------|
| Definition | ELMo produces contextual word embeddings by learning from the internal states of a two-layer bidirectional LSTM trained as a language model.| BERT produces contextual word embeddings, where the vector representation for a word depends on the entire context in which it is used.|
| Type of embeddings | Contextual word embeddings | Contextual word embeddings |
| Dimensions | 1024 | 768 |
| Training method | Bidirectional LSTM	| Transformer | 
| Advantages | Can capture long-range dependencies in text.	| Can be fine-tuned for specific tasks.| 
| Disadvantages | Can be computationally expensive to train. | Can be difficult to interpret. |
| Effectiveness | More effective than traditional word embeddings in tasks that require understanding the context of a word	| More effective than traditional word embeddings and ELMo embeddings in a variety of natural language processing tasks | 

## Sentence/Document Embeddings (operates at a Higher Granularity)
| Model | Description|
| ------ | --------- |
| Doc2Vec | An extension of Word2Vec that generates a dense vector representation for a whole document or a paragraph.|
|Sentence-BERT| An adaptation of the BERT model to produce embeddings for full sentences.|
| Universal Sentence Encoder | A model trained to encode sentences into high dimensional vectors that can be used for various downstream tasks.| 

## Positional Embeddings (encode Position of Words in a Sequence)
| Type | Description | Used in | 
| ---- | ------------ | ------ | 
| Absolute Positional Embeddings | Encode the absolute position of each word in a sentence, thus preserving order information.	| Transformer, BERT, RoBERTa | 
| Relative Positional Embeddings | Encode relative positions between pairs of words, i.e., the distance between two words, instead of their absolute positions.	| Transformer-XL, T5 | 
| Rotary Positional Embeddings/RoPE (Rotary Positional Encoding)	| A type of relative positional encoding that employs rotation operations to encode relative positions. | Routing Transformer, RoFormer | 


## Relative Embeddings

- Relative positional embeddings are a way of representing the relative position of words in a sequence. They were introduced in the paper "Self-Attention with Relative Position Representations" and have since been used in a number of NLP models, including Transformer-XL and T5. Relative positional embeddings have been shown to be effective in modeling long-range dependencies between words in a sequence, and they have been influential in the development of NLP models.
- Let’s look at an example to clarify the concept. Consider the sentence Sunil threw the mango to Linus.”
- The relative positions of the word “mango” with respect to other words would be:
    - “Sunil”: +3
    - “threw”: +2
    - “the”: +1
    - “mango”: 0
    - “to”: -1
    - “Linus”: -2
- The above example is a high-level overview of how the Transformer model works, but it is not the complete picture. Let's take a closer look at how it actually works.
    -  Each position in a sequence is represented by a vector in the embedding matrix, similar to word embeddings.
    - For all pairs of positions $i$ and $j$ in the input sequence, the difference between their positions (i.e., $i - j$) is calculated.
    - The difference is used to index the relative position embedding vector in the embedding matrix. If the difference is $d$, then the $d-th$ row of the embedding matrix is used.
    - The relative position embedding vector is combined with the attention scores, including the normal (non-relative)  attention scores.
    - Relative positional embeddings are more generalizable to sequence lengths that weren’t seen during training than traditional absolute positional embeddings. This is especially useful for tasks that involve very long sequences of text.
