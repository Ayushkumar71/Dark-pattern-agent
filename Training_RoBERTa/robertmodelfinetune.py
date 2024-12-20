

import pandas as pd

import torch
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import Trainer, TrainingArguments
from transformers import RobertaForSequenceClassification, RobertaTokenizer




# Checking if torch is using GPU acceleration or not 
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")










# Loading the dataset
df = pd.read_csv("./Fine Tuned Code/cleaned_file.csv")

# Get unique categories and prepare mappings
fine = df['category'].unique().tolist()
fine = [s.strip() for s in fine]
num_fine = len(fine)
id2fine = {id: fine for id, fine in enumerate(fine)}
fine2id = {fine: id for id, fine in enumerate(fine)}

# Drop unnecessary column and map categories to label ids
df.drop('label', axis=1, inplace=True)
df['labels'] = df.category.map(lambda x: fine2id[x.strip()])

# Tokenizing using Roberta tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", max_length=512)

# Load Roberta model for sequence classification
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=num_fine, id2label=id2fine, label2id=fine2id)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare train, validation, and test datasets
SIZE = df.shape[0]
train_texts = list(df.text[:SIZE//2])
val_texts = list(df.text[SIZE//2:(3*SIZE)//4])
test_texts = list(df.text[(3*SIZE)//4:])
train_labels = list(df.labels[:SIZE//2])
val_labels = list(df.labels[SIZE//2:(3*SIZE)//4])
test_labels = list(df.labels[(3*SIZE)//4:])

# Tokenize the datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True) if len(val_texts) > 0 else None
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Define custom dataset class
class Dataloader(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataloaders for training, validation, and testing
train_dataloader = Dataloader(train_encodings, train_labels)
val_dataloader = Dataloader(val_encodings, val_labels)
test_dataloader = Dataloader(test_encodings, test_labels)


def compute_metrics(pred):
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }
    
    
training_args = TrainingArguments(
    output_dir="./TTC4908Model",
    do_train=True,
    do_eval=True,

    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    warmup_steps=100,
    weight_decay=0.01,
    logging_strategy="steps",

    logging_dir="./multi-class-logs",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    fp16=False,
    load_best_model_at_end=True,
)

trainer = Trainer(
    # the pre-trained model that will be fine-tuned
    model=model,

    # training arguments that we defined above
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    compute_metrics=compute_metrics
)


# Train the model
trainer.train() 

# Save the trained model with custom name - ran this script on google collab after failing to accomodate its computation requirements
model.save_pretrained("./dark-model")

# Save the tokenizer with custom name
tokenizer.save_pretrained("./dark-model")


