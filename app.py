"""
Image Caption Generator - Streamlit Web Application

A production-ready web interface for the Transformer-based image captioning model.
This app recreates the exact model architecture and inference pipeline from training.
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time
import os
import sys
from pathlib import Path

# Add utils and models to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.model_utils import get_model_manager, ModelManager
from utils.image_processing import ImageProcessor, display_image_info, validate_image
from utils.text_processing import VocabularyManager

# Page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }

    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f0f8ff;
        margin: 1rem 0;
    }

    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 2px solid #28a745;
        margin: 1rem 0;
    }

    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }

    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []

def setup_sidebar():
    """Setup the sidebar with model configuration and information."""
    st.sidebar.title("🔧 Model Configuration")

    # Model status
    model_manager = get_model_manager()

    if model_manager.is_initialized:
        st.sidebar.success("✅ Model Ready")

        # Display model info
        model_info = model_manager.get_model_info()

        st.sidebar.markdown("### 📊 Model Information")
        st.sidebar.write(f"**Vocabulary Size:** {model_info['vocab_size']:,}")
        st.sidebar.write(f"**Max Sequence Length:** {model_info['max_length']}")
        st.sidebar.write(f"**Embedding Dimension:** {model_info['embedding_dim']}")

        if 'total_parameters' in model_info:
            st.sidebar.write(f"**Total Parameters:** {model_info['total_parameters']:,}")
            st.sidebar.write(f"**Trainable Parameters:** {model_info['trainable_parameters']:,}")

        # Model architecture details
        with st.sidebar.expander("🏗️ Architecture Details"):
            st.write("**CNN Encoder:** InceptionV3 (pretrained)")
            st.write("**Transformer Encoder:** 1 head, 512 dim")
            st.write("**Transformer Decoder:** 8 heads, 512 dim")
            st.write("**Image Input Size:** 299×299×3")
            st.write("**Preprocessing:** Resize + Normalize [0,1]")
    else:
        st.sidebar.warning("⚠️ Model Not Initialized")

    # File upload section
    st.sidebar.markdown("### 📁 Model Files")

    weights_file = st.sidebar.file_uploader(
        "Upload Model Weights (.h5)",
        type=['h5'],
        help="Upload the trained model weights file (image_captioning_transformer_weights.h5)"
    )

    vocab_file = st.sidebar.file_uploader(
        "Upload Vocabulary (.pkl)",
        type=['pkl'],
        help="Upload the vocabulary file from training"
    )

    # Initialize model button
    if st.sidebar.button("🚀 Initialize Model"):
        initialize_model(weights_file, vocab_file)

    # Demo mode button
    if st.sidebar.button("🎮 Demo Mode (No Weights)"):
        initialize_demo_model()

    # Generation history
    if st.session_state.generation_history:
        st.sidebar.markdown("### 📋 Recent Generations")
        for i, (image_name, caption) in enumerate(reversed(st.session_state.generation_history[-5:])):
            with st.sidebar.expander(f"#{len(st.session_state.generation_history) - i}: {image_name}"):
                st.write(f"*{caption}*")

def initialize_model(weights_file=None, vocab_file=None):
    """Initialize the model with uploaded files."""
    model_manager = get_model_manager()

    with st.spinner("🔄 Initializing model..."):
        # Save uploaded files temporarily if provided
        weights_path = None
        vocab_path = None

        if weights_file:
            weights_path = f"temp_weights_{int(time.time())}.h5"
            with open(weights_path, "wb") as f:
                f.write(weights_file.read())

        if vocab_file:
            vocab_path = f"temp_vocab_{int(time.time())}.pkl"
            with open(vocab_path, "wb") as f:
                f.write(vocab_file.read())

        # Initialize model
        success = model_manager.initialize_model(weights_path, vocab_path)

        # Clean up temporary files
        if weights_path and os.path.exists(weights_path):
            os.remove(weights_path)
        if vocab_path and os.path.exists(vocab_path):
            os.remove(vocab_path)

        if success:
            st.session_state.model_initialized = True
            st.session_state.model_manager = model_manager
            st.success("✅ Model initialized successfully!")

            # Warm up model
            with st.spinner("🔥 Warming up model..."):
                model_manager.warm_up_model()

            st.rerun()
        else:
            st.error("❌ Failed to initialize model. Check your files and try again.")

def initialize_demo_model():
    """Initialize model in demo mode without weights."""
    model_manager = get_model_manager()

    with st.spinner("🔄 Creating demo model..."):
        success = model_manager.create_dummy_model()

        if success:
            st.session_state.model_initialized = True
            st.session_state.model_manager = model_manager
            st.success("✅ Demo model created!")
            st.info("💡 Demo mode active. Upload your trained weights for actual predictions.")
            st.rerun()
        else:
            st.error("❌ Failed to create demo model.")

def process_image_and_generate_caption(uploaded_file):
    """Process uploaded image and generate caption."""
    model_manager = get_model_manager()

    if not model_manager.is_initialized:
        st.error("❌ Please initialize the model first using the sidebar.")
        return

    try:
        # Display image info
        st.markdown("### 🖼️ Input Image")
        display_image_info(uploaded_file)

        # Validate image
        if not validate_image(uploaded_file):
            return

        # Process image
        with st.spinner("🔍 Processing image..."):
            image_tensor = ImageProcessor.process_for_inference(uploaded_file)

        # Generate caption
        with st.spinner("🧠 Generating caption..."):
            start_time = time.time()
            caption = model_manager.generate_caption(image_tensor)
            generation_time = time.time() - start_time

        # Display results
        st.markdown("### 📝 Generated Caption")
        st.markdown(f"""
        <div class="result-box">
            <h4>"{caption}"</h4>
            <small>Generated in {generation_time:.2f} seconds</small>
        </div>
        """, unsafe_allow_html=True)

        # Add to history
        st.session_state.generation_history.append((uploaded_file.name, caption))

        # Additional information
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Technical Details")
            st.write(f"**Processing Time:** {generation_time:.3f} seconds")
            st.write(f"**Image Size:** {ImageProcessor.get_image_dimensions(uploaded_file)}")
            st.write(f"**Model Input Size:** 299×299×3")

        with col2:
            st.markdown("#### 🔧 Model Settings")
            st.write(f"**Max Length:** {model_manager.vocab_manager.max_length}")
            st.write(f"**Vocabulary Size:** {model_manager.vocab_manager.vocabulary_size}")
            st.write(f"**Architecture:** Transformer + InceptionV3")

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Setup sidebar
    setup_sidebar()

    # Main title
    st.markdown('<h1 class="main-header">🖼️ Image Caption Generator</h1>', unsafe_allow_html=True)

    # Description
    st.markdown("""
    <div class="info-box">
        <h4>🚀 Production-Ready Image Captioning</h4>
        <p>This application uses a Transformer-based neural network with InceptionV3 CNN encoder 
        to generate natural language descriptions of images. The model architecture exactly matches 
        the training implementation for consistent results.</p>

        <p><strong>Features:</strong></p>
        <ul>
            <li>🧠 Transformer architecture with multi-head attention</li>
            <li>🖼️ InceptionV3 CNN feature extraction</li>
            <li>📝 Natural language caption generation</li>
            <li>⚡ Real-time inference</li>
            <li>🎯 Production-ready deployment</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Check model status
    model_manager = get_model_manager()

    if not model_manager.is_initialized:
        st.markdown("""
        <div class="error-box">
            <h4>⚠️ Model Not Ready</h4>
            <p>Please initialize the model using the sidebar:</p>
            <ol>
                <li>Upload your trained model weights (.h5 file)</li>
                <li>Upload your vocabulary file (.pkl file)</li>
                <li>Click "Initialize Model"</li>
            </ol>
            <p>Or click "Demo Mode" to test without trained weights (random outputs).</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # File upload section
    st.markdown("### 📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image to generate a caption. Supported formats: PNG, JPG, JPEG, BMP, TIFF"
    )

    if uploaded_file is not None:
        process_image_and_generate_caption(uploaded_file)

    # Example images section
    st.markdown("### 🎯 How it Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **🔍 Image Processing**
        - Resize to 299×299 pixels
        - Normalize pixel values [0,1]
        - Extract features with InceptionV3
        """)

    with col2:
        st.markdown("""
        **🧠 Transformer Encoding**
        - Multi-head attention on image features  
        - Contextual feature representation
        - Prepare for sequential decoding
        """)

    with col3:
        st.markdown("""
        **📝 Caption Generation**
        - Sequential word prediction
        - Attention over image features
        - Natural language output
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 <strong>Image Caption Generator</strong> | Built with Streamlit & TensorFlow</p>
        <p>Transformer architecture with InceptionV3 CNN encoder for state-of-the-art image captioning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
