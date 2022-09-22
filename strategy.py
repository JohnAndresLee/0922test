import random
import sys, cntk
import numpy as np


with open('words_250000_train.txt') as f:
    words = f.read().splitlines()
random.shuffle(words)
# words = words[:50000]

# During training, the model will only see words below this index.
# The remainder of the words can be used as a validation set.

train_num = int(len(words) * 0.8)
print('Training with {} WordNet words'.format(train_num))

MAX_NUM_INPUTS = max([len(i) for i in words[:train_num]])
EPOCH_SIZE = train_num
NUM_EPOCHS = 15
BATCH_SIZE = np.array([len(i) for i in words[:train_num]]).mean()
print('Max word length: {}, average word length: {:0.1f}'.format(MAX_NUM_INPUTS, BATCH_SIZE))

from cmath import inf


class HangmanPlayer:
    def __init__(self, word, model, lives=10):
        self.original_word = word
        self.full_word = [ord(i)-97 for i in word]
        self.letters_guessed = set([])
        self.letters_remaining = set(self.full_word)
        self.lives_left = lives
        self.obscured_words_seen = []
        self.letters_previously_guessed = []
        self.guesses = []
        self.correct_responses = []
        self.z = model
        return

    def encode_obscured_word(self):
        word = [i if i in self.letters_guessed else 26 for i in self.full_word]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word):
            obscured_word[i, j] = 1
        return(obscured_word)

    def encode_guess(self, guess):
        encoded_guess = np.zeros(26, dtype=np.float32)
        encoded_guess[guess] = 1
        return(encoded_guess)

    def encode_previous_guesses(self):
        # Create a 1 x 26 vector where 1s indicate that the letter was previously guessed
        guess = np.zeros(26, dtype=np.float32)
        for i in self.letters_guessed:
            guess[i] = 1
        return(guess)

    def encode_correct_responses(self):
        # To be used with cross_entropy_with_softmax, this vector must be normalized
        response = np.zeros(26, dtype=np.float32)
        for i in self.letters_remaining:
            response[i] = 1.0
        response /= response.sum()
        return(response)

    def store_guess_and_result(self, guess):
        # Record what the model saw as input: an obscured word and a list of previously-guessed letters
        self.obscured_words_seen.append(self.encode_obscured_word())
        self.letters_previously_guessed.append(self.encode_previous_guesses())

        # Record the letter that the model guessed, and add that guess to the list of previous guesses
        self.guesses.append(guess)
        self.letters_guessed.add(guess)

        # Store the "correct responses"
        correct_responses = self.encode_correct_responses()
        self.correct_responses.append(correct_responses)

        # Determine an appropriate reward, and reduce # of lives left if appropriate
        if guess in self.letters_remaining:
            self.letters_remaining.remove(guess)

        if self.correct_responses[-1][guess] < 0.00001:
            self.lives_left -= 1
        return

    def run(self):
        # Play a game until we run out of lives or letters
        while (self.lives_left > 0) and (len(self.letters_remaining) > 0):
            ans = np.squeeze(self.z.eval({self.z.arguments[0]: np.array(self.encode_obscured_word()),
                                                      self.z.arguments[1]: np.array(self.encode_previous_guesses())}))
            for i in self.letters_guessed:
                ans[i] = -inf
            guess = np.argmax(ans)
            
            self.store_guess_and_result(guess)

        # Return the observations for use in training (both inputs, predictions, and losses)
        return(np.array(self.obscured_words_seen),
               np.array(self.letters_previously_guessed),
               np.array(self.correct_responses))

    def show_words_seen(self):
        for word in self.obscured_words_seen:
            print(''.join([chr(i + 97) if i != 26 else ' ' for i in word.argmax(axis=1)]))

    def show_guesses(self):
        for guess in self.guesses:
            print(chr(guess + 97))

    def play_by_play(self):
        print('Hidden word was "{}"'.format(self.original_word))
        for i in range(len(self.guesses)):
            word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in self.obscured_words_seen[i].argmax(axis=1)])
            print('Guessed {} after seeing "{}"'.format(chr(self.guesses[i] + 97),
                                                        word_seen))

    def evaluate_performance(self):
        # Assumes that the run() method has already been called
        ended_in_success = self.lives_left > 0
        letters_in_word = set([i for i in self.original_word])
        correct_guesses = len(letters_in_word) - len(self.letters_remaining)
        incorrect_guesses = len(self.guesses) - correct_guesses
        return(ended_in_success, correct_guesses, incorrect_guesses, letters_in_word)

def create_LSTM_net(input_obscured_word_seen, input_letters_guessed_previously):
    with cntk.layers.default_options(initial_state = 0.1):
        lstm_outputs = cntk.layers.Recurrence(cntk.layers.LSTM(MAX_NUM_INPUTS))(input_obscured_word_seen)
        final_lstm_output = cntk.ops.sequence.last(lstm_outputs)
        combined_input = cntk.ops.splice(final_lstm_output, input_letters_guessed_previously)
        dense_layer = cntk.layers.Dense(26, name='final_dense_layer')(combined_input)
        return(dense_layer)

input_obscured_word_seen = cntk.ops.input_variable(shape=27,
                                                   dynamic_axes=[cntk.Axis.default_batch_axis(),
                                                                 cntk.Axis.default_dynamic_axis()],
                                                   name='input_obscured_word_seen')
input_letters_guessed_previously = cntk.ops.input_variable(shape=26,
                                                           dynamic_axes=[cntk.Axis.default_batch_axis()],
                                                           name='input_letters_guessed_previously')

z = create_LSTM_net(input_obscured_word_seen, input_letters_guessed_previously)


# define loss and displayed metric
input_correct_responses = cntk.ops.input_variable(shape=26,
                                                  dynamic_axes=[cntk.Axis.default_batch_axis()],
                                                  name='input_correct_responses')
pe = cntk.losses.cross_entropy_with_softmax(z, input_correct_responses)
ce = cntk.metrics.classification_error(z, input_correct_responses)

learning_rate = 0.1
lr_schedule = cntk.learners.learning_rate_schedule(learning_rate, cntk.UnitType.minibatch)
momentum_time_constant = cntk.learners.momentum_as_time_constant_schedule(BATCH_SIZE / -np.log(0.9))
learner = cntk.learners.fsadagrad(z.parameters,
                                  lr=lr_schedule,
                                  momentum=momentum_time_constant,
                                  unit_gain = True)
trainer = cntk.Trainer(z, (pe, ce), learner)
progress_printer = cntk.logging.progress_print.ProgressPrinter(freq=EPOCH_SIZE, tag='Training')

total_samples = 0

for epoch in range(NUM_EPOCHS):
    i = 0
    while total_samples < (epoch+1) * EPOCH_SIZE:
        word = words[i]
        i += 1

        other_player = HangmanPlayer(word, z)
        words_seen, previous_letters, correct_responses = other_player.run()

        trainer.train_minibatch({input_obscured_word_seen: words_seen,
                                 input_letters_guessed_previously: previous_letters,
                                 input_correct_responses: correct_responses})
        total_samples += 1
        progress_printer.update_with_trainer(trainer, with_metric=True)

    progress_printer.epoch_summary(with_metric=True)

model_filename = './hangman_model50000-6-while.dnn'
z.save(model_filename)