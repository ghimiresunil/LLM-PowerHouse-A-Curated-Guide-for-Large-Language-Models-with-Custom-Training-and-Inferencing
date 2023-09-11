from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
import argparse
import numpy as np

nltk.download('stopwords')


def preprocessing(corpus):
  stop_words = set(stopwords.words("english"))
  training_data = []
  sentences = corpus.split(".")

  for i in range(len(sentences)):
    sentences[i] = sentences[i].strip()
    words = sentences[i].split()
    training_words =  [word.strip(string.punctuation) for word in words if word not in stop_words]
    lowered_training_words = [word.lower() for word in training_words]
    training_data.append(lowered_training_words)

  return training_data


def generate_vocab_index(sentences):
  data = list(set(word for sentence in sentences for word in sentence))
  data = sorted(data)
  vocab = {}
  for i in range(len(data)):
    vocab[data[i]] = i
  return vocab ,data


def generate_training_data(window_size, sentence_token, vocab_index):
    X_train = []
    y_train = []
    for sentence in sentence_token:
        for i in range(len(sentence)):
            center_word = [0 for x in range(len(vocab_index))]
            center_word[vocab_index[sentence[i]]] = 1
            context = [0 for x in range(len(vocab_index))]

            for j in range(i-window_size,i+window_size+1):
                if i!=j and j>=0 and j<len(sentence):
                    context[vocab_index[sentence[j]]] += 1
            X_train.append(context)
            y_train.append(center_word)
    return X_train, y_train


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class CBOWWord2Vec:
    def __init__(self, words, word_index):
        self.N = 10
        self.alpha = 0.001
        self.words = words
        self.word_index = word_index
        self.initialize_weight()

    def initialize_weight(self):
        self.W = np.random.uniform(-0.8, 0.8, (len(self.words), self.N))
        self.W1 = np.random.uniform(-0.8, 0.8, (self.N, len(self.words)))

    def feed_forward(self, X):
        self.h = np.dot(self.W.T, X).reshape(self.N,1)
        self.u = np.dot(self.W1.T, self.h)
        self.y = softmax(self.u)
        return self.y

    def backpropagate(self, x, t):
        e = self.y - np.asarray(t).reshape(len(self.words),1)
        dLdW1 = np.dot(self.h,e.T)
        X = np.array(x).reshape(len(self.words),1)
        dLdW = np.dot(X, np.dot(self.W1,e).T)
        self.W1 = self.W1 - self.alpha*dLdW1
        self.W = self.W - self.alpha*dLdW

    def train(self,epochs,x_train, y_train):
        for x in range(1,epochs):
          self.loss = 0
          for j in range(len(x_train)):
            self.feed_forward(x_train[j])
            self.backpropagate(x_train[j],y_train[j])
            C = 0
            for m in range(len(self.words)):
              if(y_train[j][m]):
                self.loss += -1*self.u[m][0]
                C += 1
            self.loss += C*np.log(np.sum(np.exp(self.u)))
          print("epoch ",x, " loss = ",self.loss)
          self.alpha *= 1/( (1+self.alpha*x) )

    def predict(self, context_words, number_of_predictions):
        context_vector = [0 for i in range(len(self.words))]
        for word in context_words:
            if word in self.words:
                index = self.word_index[word]
                context_vector[index] = 1
        
        if context_vector and any(item == 1 for item in context_vector):
            prediction = self.feed_forward(context_vector)
            output = {}
            for i in range(len(self.words)):
                output[prediction[i][0]] = i
    
            top_focus_words = []
            for k in sorted(output, reverse=True):
                top_focus_words.append(self.words[output[k]])
                if len(top_focus_words) >= number_of_predictions:
                    break
    
            return top_focus_words
        else:
            print("Context words not found in dictionary")


def main(fname, window_size, epochs):
    with open(fname,"r") as file:
        data = file.read().replace('\n', '')
    
    sentence_token = preprocessing(data)
    vocab_index, data = generate_vocab_index(sentence_token)
    
    X_train, y_train = generate_training_data(
        window_size= window_size,
        sentence_token= sentence_token,
        vocab_index= vocab_index
    )
    
    w2v = CBOWWord2Vec(
        words= data,
        word_index = vocab_index
    )
    
    w2v.train(
        epochs= epochs,
        x_train= X_train,
        y_train= y_train
    )
    
    while(True):
        embedding_word = input("Please Enter the word you are looking to see the skipgram prediction")
        neighbour_words = w2v.predict(embedding_word,3)
        print(neighbour_words)


def parser_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--file_dir",
        type= str,
        required= True,
        help="Enter the .txt file for training skipgram model"
    )
    
    parser.add_argument(
        "--window_size",
        type = int,
        default= 2,
        help= "Enter the window size for which the skipgram must work"
    )
    
    parser.add_argument(
        "--epochs",
        type= int,
        default= 10,
        help= "Enter epochs to train the model."
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parser_args()
    
    main(args.file_dir, args.window_size, args.epochs)
