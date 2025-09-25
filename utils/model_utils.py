"""
Model loading and inference utilities.
Handles model initialization, weight loading, and caption generation.
"""

import tensorflow as tf
import numpy as np
import streamlit as st
import os
from typing import Optional, Tuple
import pickle

from models.model_architecture import build_model
from utils.text_processing import VocabularyManager

# Model configuration constants (matching training exactly)
MAX_LENGTH = 40
VOCABULARY_SIZE = 10000
EMBEDDING_DIM = 512
UNITS = 512

class ModelManager:
    """
    Manages model loading, initialization, and inference.
    """

    def __init__(self):
        self.model = None
        self.vocab_manager = None
        self.is_initialized = False

    def initialize_model(self, weights_path: Optional[str] = None, vocab_path: Optional[str] = None) -> bool:
        """
        Initialize the model with exact same architecture and parameters as training.

        Args:
            weights_path: Path to saved model weights
            vocab_path: Path to saved vocabulary

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize vocabulary manager
            self.vocab_manager = VocabularyManager(VOCABULARY_SIZE, MAX_LENGTH)

            # Try to load vocabulary, fallback to demo if not available
            if vocab_path and os.path.exists(vocab_path):
                vocab_loaded = self.vocab_manager.load_vocabulary(vocab_path)
            else:
                st.warning("Vocabulary file not found. Using demo vocabulary. "
                          "For best results, provide the vocabulary file from training.")
                vocab_loaded = False

            if not vocab_loaded:
                self.vocab_manager.create_demo_setup()

            # Build model with exact training architecture
            st.info("Building model architecture...")
            self.model = build_model(
                vocab_size=self.vocab_manager.vocabulary_size,
                max_length=MAX_LENGTH,
                embedding_dim=EMBEDDING_DIM,
                units=UNITS
            )

            # Load weights if provided
            if weights_path and os.path.exists(weights_path):
                st.info("Loading model weights...")
                self.model.load_weights(weights_path)
                st.success("Model weights loaded successfully!")
            else:
                st.warning("No weights file provided. Using untrained model. "
                          "For actual predictions, provide the trained weights file.")

            self.is_initialized = True
            return True

        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return False

    def create_dummy_model(self):
        """Create a model instance for demonstration when weights aren't available."""
        try:
            self.vocab_manager = VocabularyManager(VOCABULARY_SIZE, MAX_LENGTH)
            self.vocab_manager.create_demo_setup()

            self.model = build_model(
                vocab_size=self.vocab_manager.vocabulary_size,
                max_length=MAX_LENGTH,
                embedding_dim=EMBEDDING_DIM,
                units=UNITS
            )

            self.is_initialized = True
            st.info("Demo model created. This will generate random outputs since no trained weights are loaded.")
            return True

        except Exception as e:
            st.error(f"Error creating demo model: {str(e)}")
            return False

    def generate_caption(self, image_tensor: tf.Tensor) -> str:
        """
        Generate caption for an image using the exact same inference logic as training.

        Args:
            image_tensor: Preprocessed image tensor with batch dimension

        Returns:
            Generated caption string
        """
        if not self.is_initialized or self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        try:
            # Extract image features using CNN encoder
            img_embed = self.model.cnn_model(image_tensor)

            # Encode features using transformer encoder
            img_encoded = self.model.encoder(img_embed, training=False)

            # Generate caption word by word (same as training inference)
            y_inp = '[start]'

            for i in range(MAX_LENGTH - 1):
                # Tokenize current sequence
                tokenized = self.vocab_manager.tokenizer([y_inp])[:, :-1]
                mask = tf.cast(tokenized != 0, tf.int32)

                # Get prediction from decoder
                pred = self.model.decoder(
                    tokenized, img_encoded, training=False, mask=mask
                )

                # Get the word with highest probability at current position
                pred_idx = np.argmax(pred[0, i, :])
                pred_word = self.vocab_manager.idx2word(pred_idx).numpy().decode('utf-8')

                # Stop if we hit the end token
                if pred_word == '[end]':
                    break

                # Add predicted word to sequence
                y_inp += ' ' + pred_word

            # Clean up the caption (remove start token)
            caption = y_inp.replace('[start] ', '')
            return caption

        except Exception as e:
            st.error(f"Error generating caption: {str(e)}")
            return "Error generating caption"

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        info = {
            'initialized': self.is_initialized,
            'vocab_size': VOCABULARY_SIZE,
            'max_length': MAX_LENGTH,
            'embedding_dim': EMBEDDING_DIM,
            'units': UNITS,
            'model_loaded': self.model is not None,
            'vocab_loaded': self.vocab_manager is not None and self.vocab_manager.is_ready()
        }

        if self.model is not None:
            # Get model parameter count
            total_params = self.model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
            info['total_parameters'] = total_params
            info['trainable_parameters'] = trainable_params

        return info

    def warm_up_model(self):
        """Warm up the model with a dummy prediction to improve subsequent inference speed."""
        if not self.is_initialized:
            return False

        try:
            # Create dummy image tensor
            dummy_image = tf.random.normal((1, 299, 299, 3))

            # Run a dummy prediction
            _ = self.generate_caption(dummy_image)

            return True
        except Exception as e:
            st.warning(f"Model warm-up failed: {str(e)}")
            return False


def save_model_components(model, tokenizer, save_dir: str):
    """
    Save model weights and vocabulary for later use.

    Args:
        model: Trained ImageCaptioningModel
        tokenizer: Configured TextVectorization layer
        save_dir: Directory to save components
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights
    weights_path = os.path.join(save_dir, "model_weights.h5")
    model.save_weights(weights_path)

    # Save vocabulary
    vocab_path = os.path.join(save_dir, "vocabulary.pkl")
    vocabulary = tokenizer.get_vocabulary()
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocabulary, f)

    print(f"Model components saved to {save_dir}/")
    return weights_path, vocab_path


def load_model_from_files(weights_path: str, vocab_path: str) -> Tuple[tf.keras.Model, VocabularyManager]:
    """
    Load complete model from saved files.

    Args:
        weights_path: Path to model weights
        vocab_path: Path to vocabulary

    Returns:
        Tuple of (model, vocabulary_manager)
    """
    # Initialize vocabulary
    vocab_manager = VocabularyManager(VOCABULARY_SIZE, MAX_LENGTH)
    vocab_manager.load_vocabulary(vocab_path)

    # Build and load model
    model = build_model(VOCABULARY_SIZE, MAX_LENGTH, EMBEDDING_DIM, UNITS)
    model.load_weights(weights_path)

    return model, vocab_manager


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return model_manager
