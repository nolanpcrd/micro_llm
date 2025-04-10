import re
import random
import numpy as np


class Training:
    def __init__(self, data_file, learning_rate=0.01, epochs=1000):
        self.file = data_file
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.uniform(-0.1, 0.1, (800, 1322))
        self.bias = np.zeros((1, 1322))
        self.weights_output = None
        self.bias_output = None
        self.unique_words = None
        self.vector_size = 16
        self.embeddings = {}
        self.training_text = ''
        self.word_to_index = None
        self.index_to_word = None
        self.tokens = None
        self.words = []
        self.read_file()
        self.normalize_data()
        self.split_words()
        self.tokenize()
        self.init_embeddings()

    def read_file(self):
        with open(self.file, 'r') as file:
            self.training_text = file.read()

    def normalize_data(self):
        self.training_text = self.training_text.lower()
        self.training_text = re.sub(r'[\'\"«».,!?;:()\[\]]', '', self.training_text)
        return self.training_text.strip()

    def split_words(self):
        self.words = self.training_text.split()

    def tokenize(self):
        self.unique_words = list(set(self.words))
        self.weights_output = np.random.uniform(-0.1, 0.1, (1322, len(self.unique_words)))
        self.bias_output = np.zeros((1, len(self.unique_words)))
        self.word_to_index = {word: i for i, word in enumerate(self.unique_words)}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        self.tokens = [self.word_to_index[word] for word in self.words]

    def init_embeddings(self):
        self.embeddings = {word: [random.uniform(-1, 1) for _ in range(self.vector_size)] for word in self.unique_words}

    def generate_input_output(self, input_length=50):
        combinaisons = []
        for i in range(len(self.tokens) - input_length - 1):
            input_seq = [self.tokens[i + j] for j in range(input_length)]
            output = self.tokens[i + input_length]
            combinaisons.append((input_seq, output))
        return combinaisons

    def convert_input_output(self, combinaisons):
        converted = []
        for input_seq, output in combinaisons:
            input_vecs = [self.embeddings[self.index_to_word[token]] for token in input_seq]
            output_one_hot = self.one_hot_encode(output)
            converted.append((input_vecs, output_one_hot))
        return converted

    def one_hot_encode(self, index):
        vector = np.zeros(len(self.unique_words))
        vector[index] = 1
        return vector

    def flatten_input(self, inputs):
        return np.array([np.array(input_seq).flatten() for input_seq in inputs])

    def forward_propagation(self, inputs):
        hidden_output = np.maximum(0, np.dot(inputs, self.weights) + self.bias)
        output_layer = self.softmax(np.dot(hidden_output, self.weights_output) + self.bias_output)
        return hidden_output, output_layer

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, outputs, predictions):
        return -np.sum(outputs * np.log(predictions + 1e-10)) / outputs.shape[0]

    def backward_propagation(self, inputs, hidden_output, outputs, predictions):
        d_output = predictions - outputs
        d_weights_output = np.dot(hidden_output.T, d_output)
        d_bias_output = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, self.weights_output.T)
        d_hidden[hidden_output <= 0] = 0

        d_weights = np.dot(inputs.T, d_hidden)
        d_bias = np.sum(d_hidden, axis=0, keepdims=True)

        return d_weights, d_bias, d_weights_output, d_bias_output

    def update_weights(self, d_weights, d_bias, d_weights_output, d_bias_output):
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias
        self.weights_output -= self.learning_rate * d_weights_output
        self.bias_output -= self.learning_rate * d_bias_output

    def train(self, inputs, outputs):
        for epoch in range(self.epochs):
            hidden_output, predictions = self.forward_propagation(inputs)
            loss = self.cross_entropy_loss(outputs, predictions)
            d_weights, d_bias, d_weights_output, d_bias_output = self.backward_propagation(inputs, hidden_output, outputs, predictions)
            self.update_weights(d_weights, d_bias, d_weights_output, d_bias_output)
            #if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, input_seq):
        input_vecs = [self.embeddings[self.index_to_word[token]] for token in input_seq]
        input_vecs = np.array(input_vecs).flatten()
        hidden_output = np.maximum(0, np.dot(input_vecs, self.weights) + self.bias)
        output_layer = self.softmax(np.dot(hidden_output, self.weights_output) + self.bias_output)
        return output_layer

    def save_model(self, file_path):
        np.savez(file_path,
                 weights=self.weights,
                 bias=self.bias,
                 weights_output=self.weights_output,
                 bias_output=self.bias_output,
                 embeddings=self.embeddings,
                 word_to_index=self.word_to_index,
                 index_to_word=self.index_to_word,
                 vector_size=self.vector_size)

    def load_model(self, file_path):
        data = np.load(file_path, allow_pickle=True)
        self.weights = data['weights']
        self.bias = data['bias']
        self.weights_output = data['weights_output']
        self.bias_output = data['bias_output']
        self.embeddings = data['embeddings'].item()
        self.word_to_index = data['word_to_index'].item()
        self.index_to_word = data['index_to_word'].item()
        self.index_to_word = {int(k): v for k, v in self.index_to_word.items()}
        self.vector_size = data['vector_size'].item()


training = Training("little_data.txt")
combinaisons = training.generate_input_output()
combinaisons = training.convert_input_output(combinaisons)
inputs, outputs = zip(*combinaisons)
inputs = training.flatten_input(inputs)
outputs = np.array(outputs)

training.train(inputs, outputs)
predictions = training.forward_propagation(inputs)[1]
print("Final Loss:", training.cross_entropy_loss(outputs, predictions))
training.save_model("trained_model.npz")
