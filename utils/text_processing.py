"""
Text preprocessing utilities for image captioning.
Matches the exact preprocessing used during training.
"""

import re
import tensorflow as tf
import pickle
import os
from typing import List, Optional

def preprocess_caption(text: str) -> str:
    """
    Preprocess caption text with exact same logic as training.

    Args:
        text: Raw caption text

    Returns:
        Preprocessed text with [start] and [end] tokens
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    return text


def create_tokenizer(vocabulary_path: str = None, vocabulary_size: int = 10000, max_length: int = 40):
    """
    Create and configure tokenizer with exact same settings as training.

    Args:
        vocabulary_path: Path to saved vocabulary
        vocabulary_size: Maximum vocabulary size
        max_length: Maximum sequence length

    Returns:
        Configured TextVectorization layer
    """
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=None,
        output_sequence_length=max_length
    )

    if vocabulary_path and os.path.exists(vocabulary_path):
        # Load existing vocabulary
        with open(vocabulary_path, 'rb') as f:
            vocabulary = pickle.load(f)
        tokenizer.set_vocabulary(vocabulary)
        print(f"Loaded vocabulary from {vocabulary_path}")
    else:
        print("Warning: No vocabulary file found. Tokenizer needs to be adapted to data.")

    return tokenizer


def create_lookup_layers(tokenizer):
    """
    Create word-to-index and index-to-word lookup layers.

    Args:
        tokenizer: Configured TextVectorization layer

    Returns:
        Tuple of (word2idx, idx2word) lookup layers
    """
    vocabulary = tokenizer.get_vocabulary()

    word2idx = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary
    )

    idx2word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=vocabulary,
        invert=True
    )

    return word2idx, idx2word


def save_vocabulary(tokenizer, save_path: str):
    """
    Save tokenizer vocabulary for later use.

    Args:
        tokenizer: Configured TextVectorization layer
        save_path: Path to save vocabulary
    """
    vocabulary = tokenizer.get_vocabulary()
    with open(save_path, 'wb') as f:
        pickle.dump(vocabulary, f)
    print(f"Vocabulary saved to {save_path}")


def create_demo_vocabulary():
    """
    Create a basic vocabulary for demo purposes.
    This should be replaced with the actual trained vocabulary.
    """
    # Basic vocabulary based on common image captioning words
    # This is a minimal set - the actual model needs the full training vocabulary
    demo_vocab = [
        "[UNK]", "[start]", "[end]", "a", "an", "the", "is", "are", "in", "on", "at", "with",
        "and", "or", "but", "of", "to", "from", "by", "for", "as", "was", "were", "be", "been",
        "man", "woman", "boy", "girl", "person", "people", "child", "children", "dog", "cat",
        "car", "bike", "bus", "train", "plane", "boat", "house", "building", "tree", "flower",
        "grass", "water", "sky", "cloud", "sun", "snow", "rain", "road", "street", "park",
        "standing", "sitting", "walking", "running", "playing", "holding", "wearing", "looking",
        "smiling", "jumping", "riding", "driving", "flying", "swimming", "eating", "drinking",
        "red", "blue", "green", "yellow", "black", "white", "brown", "pink", "orange", "purple",
        "big", "small", "tall", "short", "long", "round", "square", "old", "young", "new",
        "beautiful", "happy", "sad", "angry", "surprised", "excited", "tired", "hungry",
        "outside", "inside", "near", "far", "up", "down", "left", "right", "front", "back",
        "day", "night", "morning", "evening", "summer", "winter", "spring", "fall"
    ]

    # Extend to reach vocabulary size with numbered tokens
    while len(demo_vocab) < 1000:
        demo_vocab.append(f"token_{len(demo_vocab)}")

    return demo_vocab[:1000]  # Limit to reasonable size for demo


def get_demo_tokenizer(vocabulary_size: int = 10000, max_length: int = 40):
    """
    Create a demo tokenizer with basic vocabulary.

    WARNING: This is for demo purposes only. 
    For production use, load the actual trained vocabulary.
    """
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=None,
        output_sequence_length=max_length
    )

    # Create demo vocabulary
    demo_vocab = create_demo_vocabulary()

    # Extend vocabulary to reach the required size
    while len(demo_vocab) < min(vocabulary_size, 5000):  # Cap at 5000 for demo
        demo_vocab.append(f"word_{len(demo_vocab)}")

    tokenizer.set_vocabulary(demo_vocab)

    return tokenizer


class VocabularyManager:
    """
    Manages vocabulary loading and tokenization for the model.
    """

    def __init__(self, vocabulary_size: int = 10000, max_length: int = 40):
        self.vocabulary_size = vocabulary_size
        self.max_length = max_length
        self.tokenizer = None
        self.word2idx = None
        self.idx2word = None

    def load_vocabulary(self, vocabulary_path: str):
        """Load vocabulary from file."""
        try:
            self.tokenizer = create_tokenizer(vocabulary_path, self.vocabulary_size, self.max_length)
            self.word2idx, self.idx2word = create_lookup_layers(self.tokenizer)
            return True
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return False

    def create_demo_setup(self):
        """Create demo setup with basic vocabulary."""
        print("Creating demo vocabulary setup...")
        self.tokenizer = get_demo_tokenizer(self.vocabulary_size, self.max_length)
        self.word2idx, self.idx2word = create_lookup_layers(self.tokenizer)
        print("Demo vocabulary created. Note: This may not match your trained model.")

    def is_ready(self) -> bool:
        """Check if vocabulary is loaded and ready."""
        return all([self.tokenizer is not None, self.word2idx is not None, self.idx2word is not None])
