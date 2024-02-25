# Overview 

- Large Language Models (LLMs) such as GPT-3 or BERT are deep neural networks using the Transformer architecture. Being foundation models, they can be transferred to various downstream tasks through fine-tuning due to their training on extensive unsupervised and unstructured data.
- The Transformer architecture comprises two parts: encoder and decoder, which are mostly identical with a few differences (will add articles on this for more clearance). 
- The article focuses more on decoder models (like GPT-x) than encoder models (like BERT and its derivatives), due to the popularity of decoder-based models in the field of generative AI. Consequently, the term LLMs is used interchangeably with "decoder-based models."
- Given an input text “prompt”, at essence what these systems do is compute a probability distribution over a “vocabulary”—the list of all words (or actually parts of words, or tokens) that the system knows about.  The vocabulary is given to the system by the human designers.  GPT-3, for example, has a vocabulary of about 50,000 tokens. [Source](https://aiguide.substack.com/p/on-detecting-whether-text-was-generated)
- It's essential to acknowledge that LLMs have some limitations, including [hallucination](https://en.wikipedia.org/wiki/Hallucination_(artificial_intelligence)) and challenges in chain of thought reasoning (though recent advancements have been made). However, it's important to remember that LLMs were primarily trained for statistical language modeling.

``` Note: The task of language modeling is to predict the next token, given a specific context.```

# Embeddings

- In Natural Language Processing (NLP), embeddings are dense vectors representing words or sentences that capture semantic and syntactic properties. These embeddings are obtained by training models like BERT, Word2Vec, GloVe, or FastText on extensive text data and enable machine learning algorithms to process textual information. In short, embeddings convert words or sentences into compact, meaningful vectors, encapsulating their semantic meaning or semantic and syntactic properties.
- Embeddings can be either contextualized or non-contextualized. Contextualized embeddings make use of other tokens in the input to determine the embedding of each token, enabling polysemous words like "bank" to have distinct embeddings based on their context, such as "finance" or "river." In contrast, non-contextualized embeddings assign fixed embeddings to each token regardless of context, and they can be pre-trained and applied in downstream tasks. Models like Word2Vec, GloVe, and FastText provide non-contextualized embeddings, while BERT and its variants offer contextualized embeddings.
- To get the token's embedding, extract the trained model's learned weights for each word. These weights create word embeddings, representing each word as a dense vector.

# Contextualized vs. Non-Contextualized Embeddings
| Aspect | Contextualized Embeddings |  Non-Contextualized Embeddings |
| ------ | -------------------------- | ------------------------------ |
| Definition | Word representations vary based on context.	| Fixed word representations independent of context.|
| Context Dependency | Captures context-specific word meanings.	| Same representation regardless of context.|
| Example Sentence 1 "I need to deposit money in the bank" | Word representation for "bank": "bank" = [0.125, 0.532, -0.753, ...] | Word representation for "bank" = [0.432, 0.634, -0.789, ...] |
| Example Sentence 2: "I sat by the river bank and relaxed." | Word representation for "bank" = [-0.346, 0.962, -0.289, ...] | Word representation for "bank" = [0.432, 0.634, -0.789, ...] |
| Context Sensitivity	| Sensitive to the surrounding words in a sentence.	| Insensitive to the surrounding words in a sentence. | 
| Language Understanding | Suitable for tasks requiring context analysis. | May lack fine-grained language understanding. | 
| Pre-training Objective | Typically trained using language modeling. | Trained using unsupervised techniques. | 
| Popular Models | BERT, GPT, RoBERTa, etc.	| Word2Vec, GloVe, fastText, etc. |

# Use-cases of Embeddings

Using embeddings, one can execute different arithmetic operations to accomplish specific tasks.

- Word similarity can be assessed by comparing the embeddings of two words, usually with cosine similarity, which measures the cosine of the angle between two vectors. A higher cosine similarity suggests that the words are more similar in their meaning or usage.
- Word analogy tasks can be solved using vector arithmetic. For instance, to find the answer to the analogy "man is to woman as king is to what?", we perform the operation "king" - "man" + "woman" on the word embeddings, yielding the answer "queen."
- Sentence Similarity utilize the **[CLS]** token's embedding from models like BERT, designed to capture the overall meaning of the sentence. Alternatively, averaging the embeddings of all tokens in each sentence and comparing the average vectors is another approach. For sentence-level tasks, Sentence-BERT (SBERT), a modification of BERT, is often preferred due to its training to produce directly comparable sentence embeddings in the semantic space, leading to improved performance. In SBERT, both sentences are input simultaneously, enabling the model to comprehend the context of each sentence in relation to the other, resulting in more precise sentence embeddings.

# How Do LLMs Work?

- As mentioned in the Overview section, LLMs are trained to predict the next token using the preceding tokens. This autoregressive process involves feeding the current generated token back into the LLM as input to generate the subsequent one, enabling their generation capabilities.
- The initial step is to take the received prompt, tokenize it, and convert it into embeddings, which are vector representations of the input text. These embeddings are randomly initialized and learned during model training, serving as non-contextualized vector forms of the input token.
- Afterward, they perform layer-by-layer attention and feed-forward computations, resulting in assigning a number or logit to each word in the vocabulary (for decoder models like GPT-N, LLaMA, etc.) or generating contextualized embeddings (for encoder models like BERT, RoBERTa, ELECTRA, etc.).
- In decoder models, the final step involves transforming each (unnormalized) logit into a (normalized) probability distribution using the Softmax function, which determines the next word to be generated in the text.


# LLM Training Steps

At a high level, training LLMs involves the following steps:

1. Corpus Preparation: Collect a vast amount of text data, such as news articles, social media posts, or web documents.
2. Tokenization: Divide the text into separate words or subword units, referred to as tokens.
3. Embedding Generation: Usually done with a randomly initialized embedding table, through the nn.Embedding class in PyTorch. Alternatively, pre-trained embeddings like Word2Vec, GloVe, FastText, etc. can be utilized as a starting point for training. These embeddings capture the non-contextualized vector representation of the input token.
4. Neural Network Training:  Train a neural network model on the input tokens
    -  For encoder models such as BERT and its variants, their training involves learning to predict the context of a given word, which is masked, through the Masked Language Modeling task (Cloze task) and the Next Sentence Prediction objective.
    - In decoder models like GPT-x, LLaMA, etc., the training process involves predicting the subsequent token in the sequence based on the preceding context of tokens.


# Computing Similarity Between Embeddings
- Encoder models yield contextualized embeddings at their output, which can be subjected to arithmetic operations for tasks like understanding the similarity between two words or identify word analogies and many more. 
- When aiming to measure word similarity, one can utilize the contextualized embeddings of the words. Conversely, for assessing sentence similarity, one can employ the output from the **[CLS]** token or average the word embeddings of all tokens. To achieve the highest effectiveness in tasks involving sentence similarity, the favored option is to use encoder model variations such as [Sentence BERT](https://sbert.net/).
- Word or sentence similarity refers to how closely two words or sentences align in their semantic meaning.
- The following are the two most prevalent methods for gauging similarity between words or sentences (it's important to note that neither of these are considered "distance metrics").

## Dot Product Similarity
- The dot product of two vectors u and v is defined as:
$$u . v = |u||v|Cos\theta$$
- It’s perhaps easiest to visualize its use as a similarity measure when $||v||=1$, as in the diagram [Source](https://math.stackexchange.com/questions/689022/how-does-the-dot-product-determine-similarity) below where cos $\theta$ = $\frac{u.v}{||u||.||v||}$ = $\frac{u.v}{||u||}$.

![dot product of two vectors](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/assets/40186859/64f831b1-13c7-4539-8200-dbc3be3c3d95)

- Here you can see that when $\theta$ = 0 and cos $\theta$ =1, i.e., the vectors are colinear, the dot product is the element-wise product of the vectors. When $\theta$ is a right angle, and cos $\theta$ =0 , i.e. the vectors are orthogonal, the dot product is $\theta$. In general, cos $\theta$ tells you the similarity in terms of the direction of the vectors (it is -1 when they point in opposite directions). This holds as the number of dimensions is increased, and cos $\theta$ thus has important uses as a similarity measure in multidimensional space, which is why it is arguably the most commonly used similarity metric.

## Geometric Intuition
- The dot product between vectors $u$ and $v$ can be understood as projecting vector $u$ onto vector $v$ (or the reverse), and then multiplying the length of the projected $u$ $(∥u∥)$ with the length of $v$ $(∥v∥)$.
- When vector $u$ is orthogonal to vector $v$, the projection of $u$ onto $v$ results in a vector with zero length, leading to a product of zero. If you imagine all potential rotations of vector $u$ while keeping vector $v$ unchanged, the dot product provides:
    - When vectors u and v are orthogonal, the projection of $u$ onto $v$ results in a vector with a length of zero. This aligns with the idea of zero similarity.
    - The maximum value $∥u∥∥v∥$ occurs when vectors u and v are aligned in the same direction.
    - The minimum value $-∥u∥∥v∥$ occurs when vectors u and v are aligned in the opposite direction.
- The division of $u⋅v$ by the product of their magnitudes $(∥u∥∥v∥)$ constrains the output within the range of [-1, 1]. This ensures scale independence, which is the fundamental principle underlying cosine similarity.

## Cosine Similarity


$$cosine\textunderscore similarity(u,v) = \frac{u⋅v}{∥u∥∥v∥} = \frac{\sum\nolimits_{i=1}^n u_iv_i}{\sqrt{\sum\nolimits_{i=1}^nu_i^2} \sqrt{\sum\nolimits_{i=1}^nv_i^2}}$$ 

- where
    - $u$ and $v$ are the two vectors being compared.
    - . represents the dot product
    - $||u||$ and $||v||$ represent the magnitudes (or norms) of the vectors, and $n$ is the number of dimensions in the vectors.
- As previously stated, the step involving length normalization (dividing $u⋅v$ by the magnitudes of $u$ and $v$, denoted as $∥u∥∥v∥$) constrains the range to [−1, 1], thereby ensuring scale invariant.

## Cosine Similarity vs. Dot Product Similarity
| Aspect | Cosine Similarity | Dot Product Similarity|
| ------- | ----------------- | ---------------------|
| Definition | Measures the cosine of the angle between vectors. | Measures the sum of element-wise products of vectors.|
| Angle Consideration | Only considers the angle between vectors. | Considers both angle and magnitude of vectors.|
| Magnitude Consideration| Ignores magnitude, focuses on direction.|Considers both magnitude and direction.| 
| Scale Invariance	| Scale invariant, suitable for diverse data samples.| data samples.	Sensitive to magnitude, may need normalization|
| Value Range | Ranges from -1 (opposite directions) to 1 (same direction).| Ranges from negative to positive values.|
| Use Cases	| 	Often used in text/document similarity, recommendation systems.| Commonly used in machine learning and neural networks.|
| Comparison within Varying Length Data	| Well-suited for comparing data with varying lengths.| May produce different values for different magnitudes.| 
| Complexity and Implementation	| Requires normalization, potentially more operations.	| Simpler to compute, fewer operations.|

# Reasoning
- Let's begin our discussion of how reasoning works in LLMs by defining reasoning as the ability to make inferences using evidence and logic. [Source](https://arxiv.org/pdf/2302.07842.pdf)
- Reasoning can take many forms, such as commonsense reasoning or mathematical reasoning
- Similarly, there are many different ways to elicit reasoning from a model, such as chain-of-thought prompting. [Source](https://arxiv.org/abs/2201.11903)
- The extent of reasoning used by LLMs in their predictions is still unclear. It is not easy to determine how much of their final output is based on reasoning and how much is based on factual information. 

# The state of the art in retrieval-augmented generation
- In industrial settings, there is a high demand for cost-effective, privacy-preserving, and reliable solutions. Startups are particularly interested in these solutions, as they do not have the resources to invest in training models from scratch or to hire expensive talent.
- Recent research and releases of chatbots have shown that they can access and use knowledge and information that is not stored in their models. This is known as retrieval augmented generation (RAG).
- RAG allows LLMs to learn in context without the need for costly fine-tuning, making them more cost-effective to use. Companies can use a single model to process and generate responses based on new data, while also tailoring their solution to meet specific needs and staying up-to-date.
- We can accomplish this in several ways, such as iteratively calling another language model to extract the information we need.
