import re
import tqdm
import tensorflow as tf
from tensorflow.keras import layers, utils
import numpy as np

# Set the path to the dataset directory
data_dir = './datasets/' 

# Function to create training and validation datasets from the directory
def create_datasets(data_dir):
    # Creating a training dataset using text_dataset_from_directory method
    raw_train_df = tf.keras.utils.text_dataset_from_directory(
        data_dir,                        # directory path
        labels="inferred",               # auto-infer labels from subdirectories
        validation_split=0.3,            # fraction of data to reserve for validation
        subset='training',               # subset of data to use ('training' or 'validation')
        label_mode='categorical',        # one-hot encode the labels
        shuffle=True,                    # shuffle the data
        seed=42                          # seed for the random number generator
    )
    # Creating a validation dataset using text_dataset_from_directory method
    raw_val_df = tf.keras.utils.text_dataset_from_directory(
        data_dir,
        labels="inferred",
        validation_split=0.3,
        subset='validation',
        label_mode='categorical',
        shuffle=True,
        seed=42
    )
    return raw_train_df, raw_val_df

# Call create_datasets function to get training and validation datasets
raw_train_df, raw_val_df = create_datasets(data_dir)

# Printing the data. Uncomment the following line
# for text_batch, label_batch in raw_train_df.take(1):
#     for i in range(10):
#         print(f"Review: {text_batch.numpy()[i]}")
#         print(f"Label: {label_batch.numpy()[i]}")

# Function to preprocess the text data by standardizing and tokenizing it
def pre_processing(text_data):
    # Convert each item in text_data to a string
    text_data = [str(item) for item in text_data]
    max_tokens = 10000 # Set maximum number of tokens in the vocabulary
    max_len = 200 # Set maximum length of each text sequence
    
    # Define the TextVectorization layer to standardize and tokenize the text data
    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize='lower_and_strip_punctuation', # Convert all text to lowercase and remove punctuation
        split='whitespace', # Tokenize the text on whitespace
        max_tokens=max_tokens, # Set maximum number of tokens in the vocabulary
        output_mode='int', # Convert text to integer sequences
        output_sequence_length=max_len, # Set maximum length of each text sequence
        ngrams=None # Use unigrams (single tokens) only
    )

    # Fit the TextVectorization layer to the text data
    vectorize_layer.adapt(text_data)
    
    # Transform the text data into token indices
    tokenized_data = vectorize_layer(text_data)
    inverse_vocab = vectorize_layer.get_vocabulary()
    
    return tokenized_data, inverse_vocab

# Call pre_processing function to preprocess the training dataset and save it to a file
data = pre_processing(raw_train_df)
np.save('./datasets/training_df2.npy', data)
