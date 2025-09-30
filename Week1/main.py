# https://www.nltk.org/api/nltk.lm.html#nltk-language-modeling-module
import json
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.tokenize import word_tokenize

SENTENCE_LENGTH = 10

def cleanup_text(message: list) -> str:
    cleaned_message = ''

    for word in message:
        if word not in ['<s>', '</s>']:

            
            if word in '.?!,():;':
                if cleaned_message == '':
                    continue
                else:
                    cleaned_message += word
            else:
                if cleaned_message == '':
                    cleaned_message = word.capitalize()
                elif cleaned_message[-1] in '.?!':
                    cleaned_message = cleaned_message + ' ' + word.capitalize()
                else:
                    cleaned_message = cleaned_message + ' ' + word
    
    return cleaned_message

def is_word(tok: str) -> bool:
    return any(ch.isalpha() for ch in tok)

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

        while len(generated) < SENTENCE_LENGTH:

            generated = model.generate(10)
            
            # cleaned_generated = [word for word in generated if word not in ['<s>', '</s>']]
            print(f"{i+1}: before:", " ".join(generated))
            cleaned_generated = cleanup_text(generated)
            print(f"{i+1}: after:", cleaned_generated)


    print("------------------------------\n")

