# Overview 

- **Classification**
    - Accuracy: The proportion of predictions that are correct.
    - Precision: The proportion of positive predictions that are actually positive.
    - Recall: The proportion of actual positives that are correctly predicted.
    - F1 score: A harmonic mean of precision and recall.
    - Please refer the section on [Evaluation Metrics for the Classification Problem](https://github.com/ghimiresunil/LLM-PowerHouse-A-Curated-Guide-for-Large-Language-Models-with-Custom-Training-and-Inferencing/tree/articles/Articles/Evaluation%20Metrics/Classification).

- **Generative Language Models**
    - [Perplexity](https://en.wikipedia.org/wiki/Perplexity): A measure of how well a language model predicts a sequence of words.
    - [Burstiness](https://machinelearning.wtf/terms/burstiness/#:~:text=If%20a%20term%20is%20used,significant%20than%20the%20first%20appearance): A measure of how likely a language model is to generate repetitive text.
- **Machine Translation/Captioning**
    -[ BLEU (BiLingual Evaluation Understudy)](https://en.wikipedia.org/wiki/BLEU): A measure of how similar a machine translation or caption is to human-generated translations or captions.
    - [CIDEr (CIDEr: Consensus-based Image Description Evaluation)](https://arxiv.org/abs/1411.5726): A measure of how similar a machine-generated image caption is to human-generated captions.
    - [METEOR (Metric for Evaluation of Translation with Explicit ORdering)](https://en.wikipedia.org/wiki/METEOR): A measure of how similar a machine translation is to a human-generated translation, taking into account word order.
- **Text Summarization**
    - [ROUGE (Recall-Oriented Understudy for Gisting Evaluation)](https://www.aclweb.org/anthology/W04-1013.pdf): A measure of how similar a machine-generated summary is to human-generated summaries.
- **Manual Evaluation by Humans**
    - [Mean Opinion Score (MOS)](https://en.wikipedia.org/wiki/Mean_opinion_score): A measure of the overall quality of a system, typically obtained by asking human evaluators to rate the system on a scale of 1 to 5. MOS is used to evaluate a variety of NLP systems, including text generation systems, machine translation systems, text summarization systems, image generation systems, and recommendation systems.
- **NLP Benchmark Suites**
    - [GLUE (General Language Understanding Evaluation)](https://gluebenchmark.com/): A benchmark suite for evaluating the performance of NLP models on a variety of tasks, including natural language inference, sentiment analysis, and question answering.
    - [SuperGLUE (Super General Language Understanding Evaluation)](https://super.gluebenchmark.com/): A benchmark suite that is more challenging than GLUE, designed to evaluate the performance of NLP models on tasks that require reasoning and commonsense knowledge.

> Note: There is a wide variety of NLP evaluation metrics available, and the best metric to use will depend on the specific task and dataset. It is important to choose metrics that are appropriate for the task and that accurately reflect the performance of the model.


## Perplexity
- Perplexity measures how well a language model predicts the next word in a sequence, and is a common metric for evaluating language model performance.
- Perplexity is rooted in the idea of entropy, which measures the level of disorder or randomness within a system. A lower perplexity score indicates that the language model excels at predicting the next word in a given sequence, while a higher score implies decreased accuracy. In simpler terms, a lower perplexity signifies greater predictability, showcasing improved generalization and performance.
    -   Imagine you have a language model that has been trained on a dataset of news articles. You want to test how well the model can predict the next word in a new news article that it has never seen before.
    - You calculate the perplexity of the model on the new news article. A lower perplexity score means that the model is better at predicting the next word in the article.
    - Suppose the perplexity score of the model is 50. This means that, on average, the model is able to predict the next word in the article with 50% accuracy.
    - Now, you train the model on a larger dataset of news articles. You then calculate the perplexity of the model on the same new news article.
    - Suppose the perplexity score of the model is now 25. This means that the model is now able to predict the next word in the article with 75% accuracy.
- Wikipedia defines perplexity as: “a measurement of how well a probability distribution or probability model predicts a sample.”
- Perplexity, in simple terms, measures how confused or uncertain a language model is when trying to guess the next word. 
    - Imagine a language model with a perplexity of 3. When you ask it to predict the next word in the sentence 'The weather is _____,' it has three equally likely options to choose from: 'sunny,' 'rainy,' or 'cloudy.' So, a lower perplexity like 3 indicates that the model is quite certain and accurate in its predictions because there are only a few choices it needs to consider.
- Mathematically, the perplexity of a language model is defined as:
    - $PPL(P,Q) = 2^{H(P,Q)}$ where, 
        - P is the true probability distribution (the actual distribution of words in the dataset).
        - Q is the predicted probability distribution generated by the language model.
        - H(P, Q) is the cross-entropy between the true distribution P and the predicted distribution Q.
        - The cross-entropy between two probability distributions P and Q is defined as:
        - $H(P,Q)= -\sum[P(x) * log2(Q(x))]$ all possible words x in the vocabulary
        - Suppose you have a language model that predicts the next word in a sentence, and you want to calculate the perplexity of the model for the sentence "I love cats dogs." 
        - Here are the predicted probabilities (Q(x)) for the next word in the sentence:
            - Q("I") = 0.1
            - Q("love") = 0.4
            - Q("cats") = 0.3
            - Q("dogs") = 0.2
        - Let's denote Q as the predicted distribution:
            - P("I") = 0.2
            - P("love") = 0.3
            - P("cats") = 0.2
            - P("dogs") = 0.3
        - Now, we can calculate the cross-entropy H(P, Q):
            -   $H(P,Q)= -\sum[P(x) * log2(Q(x))]$ for all possible words x in the vocabulary
            - (-0.2 * log2(0.1)) + (-0.3 * log2(0.4)) + (-0.2 * log2(0.3)) + (-0.3 * log2(0.2))
        - Now, calculate the sum:
            - (-0.2 * -3.3219) + (-0.3 * -1.3219) + (-0.2 * -1.737) + (-0.3 * -2.3219)
            - 0.6644 + 0.3966 + 0.3474 + 0.6966 ≈ 2.105
        - Now, calculate perplexity:
            - Perplexity(P, Q) = 2^H(P, Q) = 2^2.105 ≈ 4.22 (rounded to two decimal places)
        - So, the perplexity of the language model for the sentence "I love cats dogs" is approximately 4.22. Lower perplexity values indicate better language model performance, as they suggest that the model's predicted probabilities are closer to the true distribution of words.
- Perplexity is the result of exponentiating the average negative log-likelihood of a sequence, using the base $e$, and can also be described as the exponent of the negative log-probability. The formula for perplexity is the exponent of mean of log likelihood of all the words in an input sequence.
    - $PPL(X) = exp\{\frac{-1}{t}\sum\nolimits_{i}^tlogp_{\theta}({x_1}|x_{<1})\}$
- Perplexity is commonly used in NLP tasks such as speech recognition, machine translation, and text generation, where the most predictable option is usually the correct answer.

- When creating standard or typical content, aiming for lower perplexity is the most reliable approach. Lower perplexity results in less randomness in the text, as larger language models strive to maximize text probability, effectively minimizing negative log-probability and, consequently, perplexity. Therefore, lower perplexity is thus desired.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

inputs = tokenizer("ABC is a startup based in New York City and Paris", return_tensors = "pt")
loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
ppl = torch.exp(loss)
print(ppl)

Output: 29.48

inputs_wiki_text = tokenizer("Generative Pretrained Transformer is an opensource artificial intelligence created by OpenAI in February 2019", return_tensors = "pt")
loss = model(input_ids = inputs_wiki_text["input_ids"], labels = inputs_wiki_text["input_ids"]).loss
ppl = torch.exp(loss)
print(ppl)

Output: 211.81
```
- For more: [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity).

### Summary of Perplexity

- What is Perplexity?
    - Perplexity is a evaluation metrics common used in natural language processing and information theory to assess how well a probability distribution predicts a sample. In the context of language models, it evaluates the uncertainty of a model in predicting the next word in a sequence.

- Why use Perplexity?
    - Perplexity serves as an inverse probability metric. A lower perplexity indicates that the model’s predictions are closer to the actual outcomes, meaning the model is more confident (and usually more accurate) in its predictions.

- How is it calculated?
    - For a probability distribution $p$ and a sequence of $N$ words $w_1$, $w_2$, ...... $w_N$: $$perplexity = p(w_1, w_2, ....., w_N)^{\frac{1}{N}}$$
    - In simpler terms, if we only consider bigrams (two-word sequences) and a model assigns a probability  to the correct next word, the perplexity would be $\frac{1}{p}$
- Where to use?
    - Language Models: To evaluate the quality of language models. A model with lower perplexity is generally considered better.
    - Model Comparison: To compare different models of different versions of the same model over a dataset.

## Burstiness
- The phenomenon of burstiness states that a term previously used in a document is more likely to be used again, making subsequent appearances less significant than the first.
- A positive correlation exists between the burstiness of a word and its semantic content, indicating that words with higher information content tend to exhibit higher burstiness.
- Burstiness serves as a measure of the predictability of a text based on the consistency of sentence length and structure throughout the piece. Similar to how perplexity assesses the predictability of words, burstiness evaluates the predictability of sentences.
- While perplexity measures the randomness or complexity of word usage, burstiness assesses the variability of sentences, encompassing their lengths, structures, and tempos. Real people tend to write in bursts and lulls, naturally alternating between long and short sentences, driven by their own verbal momentum and interest in the topic.
- Burtiness($b$) is mathematically calculated as: $$b=(\frac{\sigma_T / m_T -1}{\sigma_T / m_T + 1})$$ within the interval [-1, 1]. Therefore the hypothesis is $b_H - b_{AI} ≥ 0$, where $b_H$ is the mean burstiness of human writers and $b_{AI}$ is the mean burstiness of AI AKA a particular LLM. Corpora containing predictable and periodic dispersions of switch points exhibit burstiness values closer to -1, a characteristic commonly observed in AI-generated texts. On the other hand, corpora with less predictable switching patterns exhibit burstiness values closer to 1, a trait typically associated with human-written texts. Therefore, in the context of AI-generated text detection, the hypothesis $b_H − b_{AI} ≥ 0$ holds true, where $b_H$ represents the average burstiness of human writers and $b_{AI}$ represents the average burstiness of AI, also known as a particular large language model (LLM).
 
### Summary of Burstiness

- What is Burstiness?
    - Burstiness characterizes the phenomenon of certain terms appearing unusually frequently and repeatedly within a text. It suggests that once a word is introduced, it is more probable to reoccur within a short span."

- Why consider Burstiness?
    - Burstiness can reveal patterns or biases in text generation. For example, if an AI language model frequently repeats specific words or phrases in its output, it may indicate an overreliance on particular patterns or a lack of diverse responses.

-  How is it measured?
    - Although a universally accepted method for measuring burstiness does not exist, a common approach involves examining the distribution of terms and identifying those that occur more frequently than a typical random distribution would predict.

- Where to use?
    - Text Analysis: To understand patterns in text, e.g., to see if certain terms are being repeated unusually often.

    - Evaluating Generative Models: A language model that generates text with high burstiness may indicate an overreliance on specific patterns present in its training data or a lack of diversity in its outputs.
        - In the realm of AI and recommender systems, both perplexity and burstiness offer valuable insights into the behavior of AI models, particularly generative models like LLMs
        - Perplexity can tell us how well the model predicts or understands a given dataset.
        - Burstiness provides insights into the diversity and variability of a model's outputs. In recommender systems, where textual recommendations or descriptions are generated, perplexity can help assess the quality of those recommendations. Burstiness, on the other hand, can indicate whether the system repeatedly recommends the same or similar items.







