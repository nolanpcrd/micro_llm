import numpy as np
import re

class Run:
    def __init__(self, model_path):
        self.word_to_index = {}
        self.index_to_word = {}
        self.embeddings = {}
        self.weights = None
        self.bias = None
        self.weights_output = None
        self.bias_output = None
        self.vector_size = 0
        self.input_length = 50

        self.load_model_components(model_path)
        self.last_input = ""

    def load_model_components(self, file_path):
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

    def normalize_user_input(self, user_input):
        user_input = re.sub(r"[^a-zA-Z0-9 ]", "", user_input)
        user_input = user_input.lower().split()
        return user_input

    def convert_user_input(self, user_input):
        new_user_input = []
        for word in user_input:
            if word in self.word_to_index:
                new_user_input.append(self.word_to_index[word])
        return new_user_input

    def adjust_input_length(self, user_input, max_length):
        if len(user_input) < max_length:
            user_input = [0] * (max_length - len(
                user_input)) + user_input
        elif len(user_input) > max_length:
            user_input = user_input[-max_length:]
        return user_input

    def convert_to_embeddings(self, user_input_indices):
        zero_vector = [0.0] * self.vector_size
        embeddings_list = []
        for index in user_input_indices:
            if index == 0:
                embeddings_list.append(zero_vector)
            elif index in self.index_to_word:
                word = self.index_to_word[index]
                if word in self.embeddings:
                    embeddings_list.append(self.embeddings[word])
                else:
                    embeddings_list.append(zero_vector)
            else:
                embeddings_list.append(zero_vector)
        return embeddings_list

    def flatten_input(self, inputs):
        if not inputs or not isinstance(inputs[0], list):
            return np.zeros((1, self.input_length * self.vector_size))
        return np.array(inputs).flatten().reshape(1, -1)

    def format_user_input(self, user_input):
        user_input_words = self.normalize_user_input(user_input)
        user_input_indices = self.convert_user_input(user_input_words)
        user_input_adjusted = self.adjust_input_length(user_input_indices, self.input_length)
        user_input_embeddings = self.convert_to_embeddings(user_input_adjusted)
        user_input_flat = self.flatten_input(user_input_embeddings)
        return user_input_flat

    def forward_propagation(self, inputs):
        hidden_output = np.maximum(0, np.dot(inputs, self.weights) + self.bias)
        logits = np.dot(hidden_output, self.weights_output) + self.bias_output
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        output_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return logits, output_probs

    def apply_temperature(self, logits, temperature):
        if temperature <= 0:
            temperature = 0.01

        logits = np.array(logits, dtype=np.float64)

        if np.all(logits == logits[0]):
            num_classes = logits.shape[-1]
            return np.ones(num_classes) / num_classes

        scaled_logits = logits / float(temperature)

        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        if not np.isclose(np.sum(probs), 1.0):
            probs = probs / np.sum(probs)

        if np.isnan(probs).any():
            num_classes = logits.shape[-1]
            return np.ones(num_classes) / num_classes

        return probs

    def generate_response(self, user_input, max_length=100, temperature=0.7):
        temperature = float(temperature)
        self.last_input = user_input
        current_input_text = user_input

        response_words = []
        for i in range(max_length):
            formatted_input = self.format_user_input(current_input_text)

            logits, _ = self.forward_propagation(formatted_input)

            probs = self.apply_temperature(logits[0], temperature)
            if max(probs) < 0.01:
                print("Stopped by no confidence")
                break
            if np.isnan(probs).any() or not np.isclose(np.sum(probs), 1.0):
                probs = np.ones(len(self.word_to_index)) / len(self.word_to_index)
            predicted_index = np.random.choice(len(probs), p=probs)
            predicted_word = self.index_to_word[predicted_index]

            response_words.append(predicted_word)

            current_input_words = current_input_text.split()
            next_input_words = (current_input_words + [predicted_word])[-self.input_length:]
            current_input_text = " ".join(next_input_words)

        return " ".join(response_words)
"""
if __name__ == "__main__":
    run_model = Run("trained_model.npz")
    print("\nModel loaded. Enter 'quit' to exit.")
    while True:
        user_input = input("Vous: ")
        if user_input.lower() == "quit":
            break
        response = run_model.generate_response(user_input, temperature=1)
        print("IA:", response)
        """