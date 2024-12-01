import re

import torch
from torch.nn.functional import softmax

from transformers import RobertaTokenizer, RobertaForSequenceClassification,BertForSequenceClassification,BertTokenizer,XLNetTokenizer,AutoTokenizer,XLNetForSequenceClassification





# Specify the path to the model and tokenizer directory
model_path = "/Users/ayush/Documents/Stuff/ML Boi/Project/dark-model"


# Load the tokenizer and model for Roberta-based sequence classification
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)


# Define the maximum sequence length for tokenization
max_seq_length = 512


# Function to preprocess the input text (tokenize and encode)
def preprocess_text(text):
    # Tokenize the text and return a list of tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text, add_special_tokens=True, max_length=max_seq_length, truncation=True)))
    return tokens


# Function to predict dark patterns in a given input text
def predict_dark_patterns(input_text):
    input_ids = tokenizer.encode(preprocess_text(input_text), return_tensors='pt', max_length=max_seq_length, truncation=True)

    with torch.no_grad():
        outputs = model(input_ids)

    probs = softmax(outputs.logits, dim=1).squeeze()
    predicted_category = torch.argmax(probs).item()

    return predicted_category, probs[predicted_category].item()


# Function to count dark patterns in the text file
def count_dark_patterns(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Remove whitespaces
    text_content = re.sub(r'\s+', ' ', text_content)

    # Mapping category names to numeric labels
    category_mapping = {"Urgency": 0, "Not Dark Pattern": 1, "Scarcity": 2, "Misdirection": 3, "Social Proof": 4,
                        "Obstruction": 5, "Sneaking": 6, "Forced Action": 7}

    dark_patterns = {category: 0 for category in category_mapping}
    total_sentences = 0

    sentences = re.split(r'[.!?]', text_content)
    darkdata=[]
    for sentence in sentences:
        if not sentence.strip():
            continue

        category, _ = predict_dark_patterns(sentence)
        category_name = next(key for key, value in category_mapping.items() if value == category)

        # Exclude "Not Dark Pattern" category
        if category_name != "Not Dark Pattern":
            dark_patterns[category_name] += 1
            darkdata.append(sentence)
        total_sentences += 1

    return dark_patterns, total_sentences,darkdata



# Call the function to analyze the text in the file and print the results
result, total_sentences, darksentences = count_dark_patterns('/Users/ayush/Documents/Stuff/ML Boi/Project/dark-pattern-main/server/output.txt')


# Print the count of occurrences for each dark pattern category
for category, count in result.items():
    if category != "Not Dark Pattern":
        print(f"{category}: {count} occurrences")


# Calculate the percentage of dark pattern sentences out of the total sentences
percentage = sum(result.values()) / total_sentences * 100
print(f"Percentage of Total Dark Patterns: {percentage:.2f}%")

# Print the number of dark pattern sentences and the actual sentences
print(len(darksentences))
print(darksentences)
