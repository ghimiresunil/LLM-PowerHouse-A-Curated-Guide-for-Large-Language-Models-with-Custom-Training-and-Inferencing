# Background: Representation Learning for NLP

- At their core, neural networks construct vector-based representations of input data, known as embeddings. These embeddings capture the essence of the data's structure and meaning, serving as a foundation for diverse tasks. Whether classifying images, translating languages, or generating text, neural networks leverage these latent representations to accomplish their objectives. To achieve this, they undergo a continuous learning process, guided by feedback mechanisms that refine their ability to construct increasingly accurate and informative embeddings.
- In the world of Natural Language Processing (NLP), Recurrent Neural Networks (RNNs) are like word detectives. They carefully examine each word in a sentence, one at a time, to uncover its meaning and create a unique representation for it. Think of an RNN as a special assembly line for words: each word gets processed in order, from left to right, until a final representation is crafted for each one. These representations, like detailed word profiles, can then be used for various NLP tasks, such as understanding the overall meaning of a sentence or translating it into another language. To dive deeper into this topic, Chris Olah's highly-regarded blog offers excellent insights on [LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) and [representation learning](https://colah.github.io/posts/2014-07-NLP-RNNs-Representations/) in NLP. 
- Think of RNNs as the old way of reading a sentence, plodding through it word by word. Transformers are the new kids on the block, taking a whole different approach. They don't read one word at a time. Instead, they use a special trick called "attention" to figure out how important each word is to every other word in the sentence. Then, they blend those important bits together to understand the meaning of each word better. It's like they're looking at the whole sentence at once, not just einzelne words. This approach was pretty radical back in 2017, but now it's the hot new thing in NLP. If you want to learn more about attention, check out Lilian Weng's "Attention? Attention!", it's a great place to start.


# Enter the transformer 
- History:
    - LSTMs, GRUs and other flavors of RNNs were the essential building blocks of NLP models for two decades since 1990s.
    - Transformers (proposed in the “Attention Is All You Need” paper), introduced in 2017, challenged the need for traditional methods like recurrence and convolutions in building powerful language models.
- Back in the day, before Transformers became the new hotness, the top language models were all built with RNNs like LSTMs or GRUs. They were pretty good, but not perfect
    - They struggle with really long sequences (despite using LSTM and GRU units).
    - They are fairly slow, as their sequential nature doesn’t allow any kind of parallel computing.
- fore everyone used Transformers, LSTMs were king for language models. Let's take a quick trip down memory lane.
    - ELMo (LSTM-based): 2018
    - ULMFiT (LSTM-based): 2018
- The Transformer architecture, first proposed for machine translation by Vaswani et al. (2017), fundamentally departs from recurrent architectures. It employs an encoder-decoder framework coupled with a sophisticated attention mechanism to achieve sequence transduction.
    -  They work on the entire sequence calculating attention across all word-pairs, which let them learn long-range dependencies.
    - Architecture can be processed in parallel, making training much faster.
- The unique self-attention mechanism employed by transformers significantly enhances their representational capacity, thereby enabling them to model complex linguistic relationships more effectively. 
- The performance and parallel processing advantages of Transformers led to their widespread adoption in NLP. Their novel approach to representation learning eschews recurrence entirely, relying on an attention mechanism to determine the relative importance of all words in a sentence for each individual word. The updated representation of each word is then simply the sum of linear transformations of the features of all other words, weighted by their respective importance.
- The introduction of Transformers in 2017 marked a significant departure from the dominant sequential processing paradigm employed by RNNs in NLP. The paper's title, "Attention Is All You Need," further emphasized this shift and sparked considerable discussion within the community. For a comprehensive overview of the architecture and its implications, Yannic Kilcher's video tutorial is highly recommended.
- It took a while for Transformers to shine. Then came GPT and BERT, and suddenly everyone was talking about this "attention" thing. Here's the story
    - Attention is all you need: 2017
    - Transformers revolutionizing the world of NLP, Speech, and Vision: 2018 onwards
    - GPT (Transformer-based): 2018
    - BERT (Transformer-based): 2018
- Transformers have gone beyond just words! They're now tackling images, voices, and more, as this cool family tree of famous models shows. 

<hr>

<hr>

- And, the plots below (first plot source); (second plot source) show the timeline for prevalent transformer models:

<hr>

<hr>

- Lastly, the plot below (source) shows the timeline vs. number of parameters for prevalent transformer models:

# Transformers vs. Recurrent and Convolutional Architectures: an Overview
- In vanilla language models, words are grouped together based on their closeness. Transformers, on the other hand, make every part of the data link to every other part, like they're all paying attention to each other. This is called "self-attention." This means that as soon as it starts training, the transformer can see traces of the entire data set.
- Before transformers, AI for understanding language lagged behind other fields like computer vision. Even in the recent deep learning boom, natural language processing was kind of slow on the uptake, according to Anna Rumshisky from the University of Massachusetts, Lowell.
- The arrival of Transformers marked a turning point for NLP. Like a domino effect, they triggered a chain reaction of model advances, each surpassing the previous best in different areas.
- Imagine a news article about a celebrity couple announcing their breakup. The article states, "They decided to go their separate ways after years of dating." A vanilla language model (based on say, a recurrent architecture such as RNNs, LSTMs or GRUs) might struggle to understand who "they" refers to in this sentence. It might focus on the words "decided" and "separate" and incorrectly conclude that a band is breaking up or a political party is dividing. However, a transformer would be able to connect "they" with the previous context of the article, which established that the subject is the celebrity couple. By considering all the words and their relationships within the article, the transformer would accurately understand that the couple is the one ending their relationship. This example highlights how transformers can grasp long-range relationships between words, leading to a deeper understanding of the overall meaning of a text.





