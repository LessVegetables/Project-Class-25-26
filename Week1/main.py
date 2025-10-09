# https://www.nltk.org/api/nltk.lm.html#nltk-language-modeling-module
import json
import random

import pickle
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE, Vocabulary
from nltk.tokenize import word_tokenize

SENTENCE_LENGTH = 10

def cleanup_text(message: list) -> str:
    cleaned_message = ''

    # removing duplicate </s>
    seen_s = False
    i = 0
    while i < len(message):
        if message[i] == '</s>':
            if not seen_s:
                seen_s = True
                i += 1
            else:
                message.pop(i)

        else:
            seen_s = False
            i += 1

    for word in message:
        if word not in ['<s>', '<UNK>', '\'', '`']:     

            if word in '.?!,():;«»\"\'':
                if cleaned_message == '':
                    continue
                else:
                    cleaned_message += word
            else:
                if cleaned_message == '':
                    cleaned_message = word.capitalize()
                elif '</s>' in word: ######
                    if cleaned_message[-1] not in '.?!()':
                        ending = random.choice(".?!")
                        cleaned_message = cleaned_message + ending
                elif cleaned_message[-1] in '.?!':
                    cleaned_message = cleaned_message + ' ' + word.capitalize()
                else:
                    cleaned_message = cleaned_message + ' ' + word
    # говоря это всё можно было заменить на regex фильтр. Супер!
    return cleaned_message

def is_word(tok: str) -> bool:
    return any(ch.isalpha() for ch in tok)

def generate_sentence(model: MLE, n: int, sentence_len: int, max_sentence_len=20, random_seed=None) -> str:

    PUNCT = {'.', '?', '!', ',', '(', ')', ':', ';'}
    SPECIAL = {'<s>', '</s>', '<UNK>'}


    word_count = 0
    history = ['<s>'] * (n-1)

    def sample_token(history):
        # try a few times to avoid infinite loops on filtered tokens
        for _ in range(100):
            tok = model.generate(1, history, random_seed)
            # reject punctuation as first token
            if not (word_count == 0 and tok in '.?!,():;'):
                if tok == '<UNK>':
                    continue
                if tok == '</s>' and word_count < sentence_len:
                    continue
                return tok
        return '<UNK>'  # give up (or NONE)
    
    
    while word_count < max_sentence_len:
        if sentence_len <= word_count:
            # ends "naturaly"?
            if (history[-1][-1] in '.?!();') or (history[-1] == '</s>'):

                break

        token = sample_token(history)
        history.append(token)
        if (token not in PUNCT) and (token not in SPECIAL):
            if is_word(token):
                word_count += 1

    # print(word_count, ' '.join(history), end='\t———\t')
    message = cleanup_text(history)

    return message

    # cleaned_generated = [word for word in generated if word not in ['<s>', '</s>']]
    # print(f"{i+1}: before:", " ".join(generated))
    # cleaned_generated = cleanup_text(generated)


# Ensure the tokenizer is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# --- Step 1: Load and Process the Telegram JSON ---

# REPLACE THIS with the actual filename of your export
# json_filename = '/Users/danielgehrman/Documents/Programming/trainData/chat history with katie sep 2023 — sep 2025/result.json'
json_filename = '/Users/danielgehrman/Documents/Programming/Uni/Project-Class-25-26/Week1/my-messages.json'

all_messages = []

print("Loading chat history...")

try:
    with open(json_filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for msg in data['messages']:
            if msg:
                all_messages.append(msg)

except FileNotFoundError:
    print(f"ERROR: Could not find {json_filename}.")
    print("Make sure the JSON file is in the same folder as this script.")
    exit()

print(f"Loaded {len(all_messages)} messages. Preparing to train...")

# --- Step 2: Prepare the Data for NLTK ---

# We need to tokenize EVERY message individually.
# This relates back to your question about `[tokens]`.
# padded_everygram_pipeline wants a LIST of tokenized sentences.
tokenized_corpus = []
for message in all_messages:
    # Lowercase it and split it into words
    tokens = word_tokenize(message.lower())
    tokenized_corpus.append(tokens)


# --- Step 3: Train the N-gram Model ---

# Let's try a Trigram! Change this to 2, 3, or 4 to experiment.
# n = 2
for n in range (1, 5):
    print(f"Training a {n}-gram model...")

    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_corpus) # accepts a list of lists

    model = MLE(n)
    model.vocab._cutoff = 2
    model.fit(train_data, padded_sents)
    print("Model trained!")

    # --- Step 4: Generate Some Fun Stuff ---
    print("\n--- Generated Conversation ---")

    # Let's generate 5 different attempts
    for i in range(5):
        # Generate 10 words this time, it's more fun with chat data

        generated = generate_sentence(model, n, SENTENCE_LENGTH)
        print(f"{i+1}:", generated)


    print("------------------------------\n")