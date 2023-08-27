# Overview
- This article focuses the significance of tokenization in NLP and how it helps machines to learn the meaning of words and phrases.
- Teaching machines to read and understand text is a challenging task, but it is possible with the right approach. The goal is to enable them to read and understand the meaning of text.
- To process text, maclibrarieshines first need to tokenize it, which means dividing the text into smaller units called tokens.
- To be used by language models like BERT, text must first be tokenized, which is the process of dividing it into smaller units called tokens.
- The ability of models to understand the meaning of text is still being investigated, but it is thought that they learn syntactic knowledge at lower levels of the neural network and semantic knowledge at higher levels.[Source](https://hal.inria.fr/hal-02131630/document)
- Instead of treating text as a single string, it can be broken down into a list of individual words, called tokens. This is called tokenization.

# Tokenization

Tokenization is the process of splitting a text into smaller units, called tokens, which can be words, subwords, or characters. It is one of the first steps in an NLP pipeline.

| Tokenization Type | Description | Advantages | Disadvantages | Example |
| ----------------- | ----------- | ----------- | ------------ | -------- |
| Word-based tokenization	| Breaks a text into words based on a delimiter, such as a space or punctuation mark.	| Simple and efficient.	| Large vocabulary size and many OOV tokens, which can lead to a heavier model and loss of information.	| "This is a sentence." → ["This", "is", "a", "sentence"].|
| Character-based tokenization	| Splits the raw text into individual characters.	| Very small vocabulary and no or very few OOV tokens.	| Results in very long sequences and characters do not carry as much meaning as words.	| "This is a sentence." → ["T", "h", "i", "s", " ", "i", "s", " ", "a", " ", "s", "e", "n", "t", "e", "n", "c", "e"]. |
| Subword-based tokenization | Splits words into smaller units, called subwords.	| Reduces the vocabulary size and number of OOV tokens.	| More complex and computationally expensive than word-based tokenization.	| "This is a sentence." → ["This", "is", "a", "sent", "enc", "e"]. |

# Why Tokenization
- Tokenization helps split unstructured data or text into chunks of information.
- Tokenization helps split unstructured data or text into chunks of information.
- Chunks represent the discrete elements whose occurrences in a corpus document can be represented as a vector of the corresponding document. 
- The unstructured data can be represented as a numerical data structure which can be fed directly to machine learning algorithm.

# Tokenization: How Tokenization works?
- Tokenization is the process of breaking text into smaller units called tokens. There are many different ways to tokenize text, but some common methods include removing spaces, adding a split character between words, or simply breaking the input sequence into separate words. The image you provided shows a visualization of the tokenization process. 
- We can use one of these methods to tokenize the text, which means dividing it into smaller units called tokens. These tokens can then be used as input to the model.
- To learn the relationships between words in a sequence of text, we need to represent the text as a vector.
- We use a vector representation of text instead of hard coding grammatical rules, as this is more scalable and can be applied to any language. This is because the complexity of hard coding grammatical rules would be exponential, as it would need to be done for each language.
- The vector representation of a word encodes the word's meaning, including its syntactic and semantic properties.


# Sub-word Tokenization
- Sub-word tokenization is a technique that can be used to handle rare and out-of-vocabulary words by breaking words down into smaller units. This is because many words in a language share common prefixes or suffixes.
- Subword tokenization can help handle out-of-vocabulary (OOV) words by breaking them down into smaller units, such as "any" and "body" for the word "anybody". This approach can reduce the model size and improve efficiency, while also enabling the model to handle a broader range of vocabulary.
- The choice of subword tokenization algorithm depends on the NLP task. Different algorithms have different strengths and weaknesses, so it is important to choose the one that is best suited for the task at hand.
    - Example: 
        - The word "refactoring" can be split into the subwords "re", "factor", and "ing".
        - The word "uninteresting" can be split into the subwords "un", "inter", "est", and "ing".
        - The word "outperform" can be split into the subwords "out", "per", "form", and "ing".

# Byte Pair Encoding (BPE)
- BPE is a subword tokenization algorithm that breaks down words into smaller units by merging the most common pairs of characters. It has been used in GPT-2 and RoBERTa, which are large language models.
- The algorithm starts by creating subwords from individual characters. It then repeatedly replaces the most frequent pairs of characters in the text with a new subword, which is represented by an unused byte.
- The process is repeated for a specified number of times or until the desired number of sub-words is obtained.
-  It was first described in the article “[A New Algorithm for Data Compression](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)”
- The following example, taken from [Wikipedia](https://en.wikipedia.org/wiki/Byte_pair_encoding), will explain BPE.
> Suppose we have data $aaabdaaabac$ which needs to be encoded (compressed). The byte pair aa occurs most often, so we will replace it with $Z$ as $Z$ does not occur in our data. So we now have ZabdZabac where $Z = aa$. The next common byte pair is ab so let’s replace it with $Y$. We now have $ZYdZYac$ where $Z = aa$ and $Y = ab$. The only byte pair left is ac which appears as just one so we will not encode it. We can use recursive byte pair encoding to encode ZY as X. Our data has now transformed into XdXac where $X = ZY$, $Y = ab$, and $Z = aa$. It cannot be further compressed as there are no byte pairs appearing more than once. We decompress the data by performing replacements in reverse order.
- BPE is a subword-based tokenization algorithm that merges the most common pairs of characters into single tokens, while breaking down rare words into multiple tokens. 
- Consider augmenting the given words, namely `"old," "older," "“finest”," and "lowest"` with a special end token `"</w>"` attached to the end of each word. Upon calculating their occurrences in the corpus, the resulting frequency counts are as follows.

```python
# Before special end token </w>
{“old”: 7, “older”: 3, “finest”: 9, “lowest”: 4}

# After special end token </w>
{“old</w>”: 7, “older</w>”: 3, “finest</w>”: 9, “lowest</w>”: 4}
```
> To identify word boundaries, the algorithm adds the “</w>” token to the end of each word. This helps the algorithm find the most frequent character pairs by looking through each character.

- Next, we will tokenize each word into its characters, including the “</w>” token to mark the end of the word. Then, we will count the frequency of each character.

| Number | Token | Frequency |
| ------- | ----- | --------- |
| 1 | <\w> | 23 | 
| 2 | o | 14 | 
| 3 | l | 14 |
| 4 | d | 10 |
| 5 | e | 16 |
| 6 | r | 3 |
| 7 | f | 9 |
| 8 | i | 9 |
| 9 | n | 9 |
| 10 | s | 13 |
| 11 | t | 13 |
| 12 | w | 4 |

- The number of `“</w>”` tokens is equal to the number of words, which is 23. The second most frequent token is `“e”`, and there are a total of 12 different tokens.
- The next step in the BPE algorithm is to find the most frequently occurring pair of characters and merge them. This process is repeated until we reach the desired vocabulary size or the maximum number of iterations is reached.
- The goal of the BPE algorithm is to represent the corpus with the least number of tokens, which is achieved by merging the most frequently occurring byte pairs. In English, a character is the same as a byte, but this may not be the case in other languages. Once the most common byte pairs are merged, they are added to the list of tokens and the frequency of each token is recalculated. This process is repeated until the desired number of iterations is reached or the maximum token limit is reached.






