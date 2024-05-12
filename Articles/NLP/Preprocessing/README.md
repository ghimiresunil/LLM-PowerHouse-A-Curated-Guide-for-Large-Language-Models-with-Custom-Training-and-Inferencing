## Overview
- Preprocessing transforms and cleans raw text to make it easier to understand and work with efficiently.
- These techniques reduce language data complexity, improving computational performance of models, but must be applied thoughtfully for specific NLP tasks.
- They enhance computational efficiency, reduce data complexity, and often boost NLP model performance.

## Stemming
- Stemming in NLP simplifies words to their base form by removing suffixes, like turning "jumps" into "jump".
- This process reduces the number of unique words for the model to handle, which can speed up computation.
- However, stemming can be too simplistic because it doesn't consider word context, potentially leading to incorrect results.
- The Porter stemmer algorithm applies rules to generate stems without a lookup table for actual stems.
- For languages other than English, SnowballStemmer is used for stemming.

```python
# Let’s see how to use it. First, we import the necessary modules.

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
The PorterStemmer class can be imported from the nltk.stem module. Then, we instantiate a PorterStemmer object.

stemmer = PorterStemmer()

# The function stem then can be used to actually do stemming on
# words.

print(stemmer.stem("cat")) # -> cat
print(stemmer.stem("cats")) # -> cat

print(stemmer.stem("walking")) # -> walk
print(stemmer.stem("walked")) # -> walk

print(stemmer.stem("achieve")) # -> achiev

print(stemmer.stem("am")) # -> am
print(stemmer.stem("is")) # -> is
print(stemmer.stem("are")) # -> are
cat
cat
walk
walk
achiev
am
is
are

# To stem all the words in a text, we can use the PorterStemmer
# on each token producted by the word_tokenize function.

text = "The cats are sleeping. What are the dogs doing?"

tokens = word_tokenize(text)
tokens_stemmed = [stemmer.stem(token) for token in tokens]
print(tokens_stemmed)
# ['the', 'cat', 'are', 'sleep', '.', 'what', 'are', 'the', 'dog', 'do', '?']
['the', 'cat', 'are', 'sleep', '.', 'what', 'are', 'the', 'dog', 'do', '?']
```

### Lemmatization
- Lemmatization reduces words to their base form, considering the context and part of speech, unlike stemming.
- It accurately identifies lemmas, like "good" for "better", but can be computationally more expensive than stemming.

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# We’ll use the WordNetLemmatizer which leverages WordNet to
# find existing lemmas. Then, we create an instance of the
# WordNetLemmatizer class and use the lemmatize method.

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("achieve")) # -> achieve
achieve

#The lemmatizer is able to reduce the word “achieve” to its
# lemma “achieve”, differently from stemmers which reduce it to
# the non-existing word “achiev”.
```

### Stopwords
- Stopwords, common words like "is" and "the", are often excluded from text processing to reduce dataset size and enhance computational efficiency.
- Yet, in tasks like sentiment analysis, removing stopwords might impact model performance since they can convey meaningful context, warranting careful consideration before removal.

```python
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
Then, we retrieve the stopwords for the English language with stopwords.words("english"). There are 179 stopwords in total, which are words (note that they are all in lowercase) that are very common in different types of English texts.

english_stopwords = stopwords.words('english')
print(f"There are {len(english_stopwords)} stopwords in English")
print(english_stopwords[:10])
There are 179 stopwords in English
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
```

### Tokenization
- Tokenization divides text into smaller units called tokens, such as sentences or words, serving as the initial step in NLP preprocessing.
- Sentence tokenization splits paragraphs into individual sentences, while word tokenization divides sentences into individual words.

### Lowercasing
- Lowercasing converts all text to lowercase, ensuring uniformity and preventing duplication of words like "House", "house", and "HOUSE" so they're treated as the same word.

### Punctuation Removal
- Removing punctuation helps in NLP vectorization by focusing solely on word counts rather than grammatical context, enhancing the efficiency of the process.

### Spell Check and Correction
- Spell check and correction are effective in reducing typos and spelling mistakes in text data, ensuring consistency and preventing duplication of words like "speling" and "spelling" by correcting errors.

### Noise Removal
- Noise removal involves eliminating characters, digits, and text fragments such as file headers, footers, HTML, and XML tags to enhance the clarity of text analysis.

### Text Normalization
- Text normalization encompasses processes like converting text to lowercase, removing punctuation, and replacing numbers with their word equivalents to standardize text for analysis.

### Part-of-Speech Tagging
- Part-of-speech tagging assigns each word in a sentence a part of speech (noun, verb, adjective, etc.), aiding in understanding sentence structure and facilitating tasks like named entity recognition and question answering.

```python
# POS Tagging with Python
# The NLTK library provides an easy-to-use pos_tag function that # takes a text as input and returns the part-of-speech of each # token in the text.

text = word_tokenize("They refuse to go")
print(nltk.pos_tag(text))

text = word_tokenize("We need the refuse permit")
print(nltk.pos_tag(text))
[('They', 'PRP'), ('refuse', 'VBP'), ('to', 'TO'), ('go', 'VB')]
[('We', 'PRP'), ('need', 'VBP'), ('the', 'DT'), ('refuse', 'NN'), ('permit', 'NN')]

# PRP are propositions, NN are nouns, VBP are present tense 
# verbs, VB are verbs, DT are definite articles, and so on. Read # this article to see the complete list of parts-of-speech that # can be returned by the pos_tag function. In the previous 
# example, the word “refuse” is correctly tagged as verb and 
# noun depending on the context.

# The pos_tag function assigns parts-of-speech to words
# leveraging their context (i.e. the sentences they are in),
# applying rules learned over tagged corpora.

```