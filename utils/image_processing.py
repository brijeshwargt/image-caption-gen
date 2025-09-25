"""
Image processing utilities for image captioning.
Handles image loading, preprocessing, and preparation for the model.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
from typing import Union, Tuple
import io

def preprocess_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> tf.Tensor:
    """
    Preprocess image to match training preprocessing exactly.

    Args:
        image_input: Image as file path, bytes, PIL Image, or numpy array

    Returns:
        Preprocessed image tensor ready for model input
    """
    # Handle different input types
    if isinstance(image_input, str):
        # File path
        img = tf.io.read_file(image_input)
        img = tf.io.decode_jpeg(img, channels=3)
    elif isinstance(image_input, bytes):
        # Bytes from uploaded file
        img = tf.io.decode_jpeg(image_input, channels=3)
    elif isinstance(image_input, Image.Image):
        # PIL Image
        img_array = np.array(image_input.convert('RGB'))
        img = tf.constant(img_array)
    elif isinstance(image_input, np.ndarray):
        # Numpy array
        img = tf.constant(image_input)
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Ensure we have the right data type
    img = tf.cast(img, tf.float32)

    # Resize to InceptionV3 input size (299x299) - exact same as training
    img = tf.keras.layers.Resizing(299, 299)(img)

    # Normalize to [0, 1] - exact same as training
    img = img / 255.0

    return img


def load_and_preprocess_image(image_source) -> tf.Tensor:
    """
    Load and preprocess image from Streamlit file uploader or path.

    Args:
        image_source: Image from streamlit file uploader or file path

    Returns:
        Preprocessed image tensor
    """
    try:
        if hasattr(image_source, 'read'):
            # Streamlit UploadedFile object
            image_bytes = image_source.read()
            # Reset file pointer for potential reuse
            image_source.seek(0)
            return preprocess_image(image_bytes)
        elif isinstance(image_source, str):
            # File path
            return preprocess_image(image_source)
        else:
            # Direct image input
            return preprocess_image(image_source)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        raise


def prepare_image_for_model(img_tensor: tf.Tensor) -> tf.Tensor:
    """
    Prepare preprocessed image for model inference.

    Args:
        img_tensor: Preprocessed image tensor

    Returns:
        Image tensor with batch dimension added
    """
    # Add batch dimension
    img_tensor = tf.expand_dims(img_tensor, axis=0)
    return img_tensor


def display_image_info(image_source):
    """
    Display image information in Streamlit.

    Args:
        image_source: Image source (file uploader or path)
    """
    try:
        if hasattr(image_source, 'name'):
            st.write(f"**Filename:** {image_source.name}")
            st.write(f"**Size:** {image_source.size} bytes")
            st.write(f"**Type:** {image_source.type}")

            # Display image
            image = Image.open(image_source)
            st.image(image, caption=f"Input Image: {image_source.name}", use_column_width=True)

            # Reset file pointer
            image_source.seek(0)

        elif isinstance(image_source, str):
            st.write(f"**File path:** {image_source}")
            image = Image.open(image_source)
            st.image(image, caption=f"Input Image", use_column_width=True)

    except Exception as e:
        st.error(f"Error displaying image info: {str(e)}")


def validate_image(image_source) -> bool:
    """
    Validate that the image can be processed.

    Args:
        image_source: Image source to validate

    Returns:
        True if image is valid, False otherwise
    """
    try:
        # Try to load and preprocess the image
        img_tensor = load_and_preprocess_image(image_source)

        # Check dimensions
        if len(img_tensor.shape) != 3 or img_tensor.shape[-1] != 3:
            st.error("Image must be RGB with 3 channels")
            return False

        # Check if image is not empty
        if tf.reduce_sum(img_tensor) == 0:
            st.error("Image appears to be empty or corrupted")
            return False

        return True

    except Exception as e:
        st.error(f"Invalid image: {str(e)}")
        return False


def create_image_grid(images: list, captions: list = None, cols: int = 3):
    """
    Create a grid display of images in Streamlit.

    Args:
        images: List of images (PIL Images or file paths)
        captions: Optional list of captions
        cols: Number of columns in grid
    """
    if not images:
        return

    # Create columns
    columns = st.columns(cols)

    for idx, image in enumerate(images):
        col_idx = idx % cols

        with columns[col_idx]:
            # Display image
            if isinstance(image, str):
                img = Image.open(image)
            else:
                img = image

            caption = captions[idx] if captions and idx < len(captions) else f"Image {idx + 1}"
            st.image(img, caption=caption, use_column_width=True)


class ImageProcessor:
    """
    Encapsulates all image processing functionality.
    """

    @staticmethod
    def process_for_inference(image_source) -> tf.Tensor:
        """
        Complete preprocessing pipeline for model inference.

        Args:
            image_source: Image input from various sources

        Returns:
            Preprocessed image tensor ready for model
        """
        # Validate image
        if not validate_image(image_source):
            raise ValueError("Invalid image input")

        # Load and preprocess
        img_tensor = load_and_preprocess_image(image_source)

        # Prepare for model (add batch dimension)
        img_tensor = prepare_image_for_model(img_tensor)

        return img_tensor

    @staticmethod
    def get_image_dimensions(image_source) -> Tuple[int, int]:
        """Get original image dimensions before preprocessing."""
        try:
            if hasattr(image_source, 'read'):
                image = Image.open(image_source)
                image_source.seek(0)  # Reset pointer
            elif isinstance(image_source, str):
                image = Image.open(image_source)
            else:
                image = Image.open(io.BytesIO(image_source))

            return image.size  # (width, height)
        except Exception:
            return (0, 0)
