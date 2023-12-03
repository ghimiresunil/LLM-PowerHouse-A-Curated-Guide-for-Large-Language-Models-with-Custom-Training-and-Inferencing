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

## Bilingual Evaluation Understudy (BLEU)
- **Origin**: Introduced in the paper "BLUE: A Method for Automatic Evaluation of Machine Translation"
- **Definition**: A metric for evaluating the quality of machine-translated text.
- **Purpose**: To measure the similarity between machine-generated text and a set of reference translations
- **Method**: Evaluates the precision of n-grams (consecutive sequences of n words)
- **Limitation**: Primarily focuses on precision and lacks a recall component
- When evaluating machine translation, multiple characteristics are taken into account.
    - Fluency: The machine-translated text should be natural and easy to read.
    - Adequacy: The machine-translated text should accurately convey the meaning of the source text.
    - Fidelity: The machine-translated text should preserve the nuances and style of the original text as much as possible. 
    - Grammar: The machine-translated text should be grammatically correct.
    - Style: The machine-translated text should match the style of the source text.
    - Overall quality: The machine-translated text should be of overall high quality.
- Mathematically, precision for unigram word can be calculated as:
$$Precision = \frac{Number \ of \ correct \ word \ in \ Machine \ Translation}{Total \ words \ in \ machine \ translation}$$
- BLEU expands on this concept by incorporating the precision of n-grams. To prevent artificially inflated precision scores, BLEU employs a modified precision calculation.
- Mathematically, the BLEU score for n-grams is:$$BLEU = BP * exp(\sum\nolimits_{i=1}^nw_i * log(p_i))$$ 
where,
    -  *BP* is the brevity penalty (to penalize short sentence)
    - $w_i$ are the weights for each gram (usually, we give equal weight)
    - $p_i$ is the precision for each i-gram
- In its simplest form, BLEU is the ratio of matching words to the total word count in the hypothesis sentence (translation). Considering the denominator, it's evident that BLEU is a precision-oriented metric. $$p_n = \frac{\sum\nolimits_{n-gram\in hypothesis}Count_{match}(n{-}gram)}{\sum\nolimits_{n-gram\in hypothesis}Count(n{-}gram)} = \frac{\sum\nolimits_{n-gram\in hypothesis}Count_{match}}{ℓ_{hyp}^{n{-}gram}}$$
- For example, the matches in the below sample sentences are: 'the', 'guard', 'arrived', 'late', and 'because' 
    - Sentence 01: The guard arrived late because it was raining
    - Sentence 02: The guard arrived late because of the rain
    $$p_1 = \frac{5}{8}$$
    - where 5 is the matched keywords and 8 is the length of predicted or hypothesis sentence.
> Unigram matches generally assess adequacy, while longer n-gram matches capture fluency.
- Subsequently, the calculated precision values for various n-grams are aggregated using a weighted average of their logarithms. $$BLEU_N = BP.exp(\sum\nolimits_{i=1}^Nw_nlogp_n)$$
- To mitigate the shortcomings of the precision metric, a brevity penalty is incorporated or added. This penalty is zero, or 1.0, when the hypothesis sentence length aligns with the reference sentence length.
- The brevity penalty *BP* is a function of the lengths of the reference and hypothesis sentences. 
$$ BP = 
\begin{cases}   
  1  \text{ if $ℓ_{hyp} > ℓ_{ref}$}\\
  e^{1-\frac{ℓ_{ref}}{ℓ_{hyp}}} \text{ if $ℓ_{hyp} ≤ ℓ_{ref}$}
\end{cases}
$$
- The BLEU score is a numerical value between 0 and 1, with 0.6 or 0.7 representing exceptional performance. It's important to recognize that even human translations can vary, and achieving a perfect score is often unrealistic.
- Example

| Type | Sentence | Length |
| ----- | -------- | ------ |
| Reference (by human) | The guard arrived late because it was raining | ${ℓ_{ref}^{unigram}}=8$ |
| Hypothesis/Candidate (by machine) | The guard arrived late because of the rain | ${ℓ_{hyp}^{unigram}}=8$|
- we'll utilize the parameters defined in the paper, including an N-gram order of 4 and a uniform distribution for weights, resulting in $w_n=\frac{1}{4}$.$$BLEU_{N=4} = BP.exp(\sum\nolimits_{i=1}^{N=4}\frac{1}{4}logp_n) = BP *  \Pi_{n=1}^{4}p_n^{w_n} = (P_1)^\frac{1}{4}*(P_2)^\frac{1}{4}*(P_3)^\frac{1}{4}*(P_4)^\frac{1}{4}$$
- We then calculate the precision $p_n$ for the different n-grams.
- Following are the precision values for [1,5] n-grams.

|n-gram | 1-gram | 2-gram | 3-gram | 4-gram |
|-----------|--------------|----------------|----------------|----------------|
|$p_n$ | $\frac{5}{8}$ | $\frac{4}{7}$ | $\frac{3}{6}$ | $\frac{2}{5}$ | 
- Then, we calculate the brevity penalty:$$BP = e^{1-\frac{ℓ_{ref}}{ℓ_{hyp}}} = e^{1-\frac{8}{8}}$$
- Finally, we aggregate the precision values across all n-grams, which gives:
$$BLEU_{N=4} ≈ 0.5169$$
- BLEU computation is made easy with the `sacreBLEU` python package.
```python
from sacrebleu.metrics import BLEU
bleu_scorer = BLEU()

hypothesis = "The guard arrived late because it was raining"
reference = "The guard arrived late because of the rain"

score = bleu_scorer.sentence_score(
    hypothesis=hypothesis,
    references=[reference],
)

score.score/100 # sacreBLEU gives the score in percent
```
