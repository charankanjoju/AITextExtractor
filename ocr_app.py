import streamlit as st
from PIL import Image
import pytesseract
import easyocr
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="AI Text Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with dark theme and glassmorphism
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #e2e2e2;
    }
    
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Headers */
    .title-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .title {
        background: linear-gradient(120deg, #00ff87 0%, #60efff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        margin: 0;
        letter-spacing: 2px;
    }
    
    .subtitle {
        color: #a0aec0 !important;
        font-size: 1.2rem !important;
        margin-top: 1rem !important;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(120deg, #00ff87 0%, #60efff 100%);
        color: #1a1a2e !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,255,135,0.2);
    }
    
    /* File uploader */
    .uploadedFile {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(96, 239, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .uploadedFile:hover {
        border-color: #00ff87;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 15px !important;
        color: #e2e2e2 !important;
    }
    
    /* Results section */
    .results-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
        animation: slideUp 0.5s ease-out;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { 
            opacity: 0;
            transform: translateY(20px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Success/Error messages */
    .success-msg {
        background: rgba(0, 255, 135, 0.1);
        border: 1px solid rgba(0, 255, 135, 0.3);
        color: #00ff87;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .error-msg {
        background: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.3);
        color: #ff4d4d;
        padding: 1rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Set tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize EasyOCR reader in session state
if 'reader' not in st.session_state:
    st.session_state.reader = None

def extract_text_traditional(image):
    try:
        preprocessed = preprocess_image_opencv(image)
        text = pytesseract.image_to_string(preprocessed)
        return text if text.strip() else "No text detected"
    except Exception as e:
        st.error(f"Error in traditional OCR: {str(e)}")
        return "Error occurred during text extraction."

def extract_text_deep_learning(image):
    try:
        if st.session_state.reader is None:
            with st.spinner("🤖 Initializing AI model..."):
                st.session_state.reader = easyocr.Reader(['en'])

        preprocessed = preprocess_image_opencv(image)

        with st.spinner("🔍 Analyzing image with AI..."):
            results = st.session_state.reader.readtext(preprocessed)

        text = ' '.join([result[1] for result in results])
        return text if text.strip() else "No text detected"
    except Exception as e:
        st.error(f"Error in AI OCR: {str(e)}")
        return "Error occurred during text extraction. Please try again or use traditional OCR."

def preprocess_image_opencv(image):
    try:
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    except Exception as e:
        st.error(f"Error during preprocessing: {str(e)}")
        return np.array(image)
        
# Main UI
st.markdown('<div class="title-container"><h1 class="title">AI Text Extractor</h1><p class="subtitle">Transform your images into text with advanced AI technology</p></div>', unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Image")
    image_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    st.markdown("### 🛠️ Select Method")
    ocr_method = st.selectbox(
        "Choose your preferred extraction method",
        ["✨ AI-Powered OCR", "📝 Traditional OCR"],
        help="AI-Powered OCR is more accurate but may take longer"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Display and process
if image_file is not None:
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📸 Preview")
        st.image(image_file, caption="", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Open image
    image = Image.open(image_file)
    
    # Process button
    if st.button("🔍 Extract Text"):
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown("### 📄 Extracted Text")
        
        if "Traditional" in ocr_method:
            with st.spinner("⚙️ Processing with traditional OCR..."):
                text = extract_text_traditional(image)
        else:
            with st.spinner("🤖 Processing with AI..."):
                text = extract_text_deep_learning(image)
        
        # Display results
        if text and text != "No text detected":
            st.markdown('<div class="success-msg">✨ Text extracted successfully!</div>', unsafe_allow_html=True)
            st.code(text, language=None)
        else:
            st.markdown('<div class="error-msg">⚠️ No text was detected in the image.</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
else:
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 👋 Welcome!")
        st.markdown("""
        Get started by following these steps:
        1. Upload an image containing text
        2. Choose your preferred extraction method
        3. Click 'Extract Text' to begin
        """)
        st.markdown('</div>', unsafe_allow_html=True)
