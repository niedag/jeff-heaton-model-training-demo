from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
torch.cuda.is_available()

# REQUIRES BOTH TRANSFORMERS AND TORCH LIBRARIES

# Load fine-tuned distilBERT model 
model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_emotions_distilBERT_model")

# Load up the tokenizer (can be in the same directory?)
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_emotions_distilBERT_model")

# print(model)
# print(tokenizer)

# In order for the model and tokenizers to work, models and tokenizers need to be saved right after the model has been fine-tuned
# It does NOT work if you save the model and tokenizer after closing and reopening the notebook

# Emotions dataset used is 16000 param training data, 2000 test data, 2000 validation data set

text = "You are my soulmate"

# Tokenize text
token_text = tokenizer(text, return_tensors="pt", padding=True, truncation = True)

# Forward pass tokenized text through model
with torch.no_grad():
    outputs = model(**token_text)

predicted_label = torch.argmax(outputs.logits, dim=1).item()

# Print the label
print("Predicted label is: ", predicted_label)

# Labels:
# 0: Sadness
# 1: Joy
# 2: Love
# 3: Anger
# 4: Fear
# 5: Surprise
