import streamlit as st
from src.cccClassifier.pipeline.prediction import PredictionPipeline
import os
from PIL import Image

st.set_page_config(
    page_title="Chest Cancer Classification",
    page_icon="🏥",
    layout="centered"
)

st.title("Chest Cancer Classification")
st.write("Upload a chest X-ray image to classify if it's normal or shows signs of adenocarcinoma cancer.")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the file and make prediction
    if save_uploaded_file(uploaded_file):
        try:
            # Make prediction
            pipeline = PredictionPipeline(os.path.join("uploads", uploaded_file.name))
            result = pipeline.predict()
            
            # Display prediction with appropriate styling
            prediction = result[0]["image"]
            if prediction == "Normal":
                st.success(f"Prediction: {prediction}")
            else:
                st.error(f"Prediction: {prediction}")
                
            # Clean up the uploaded file
            os.remove(os.path.join("uploads", uploaded_file.name))
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Add some information about the model
st.markdown("---")
st.markdown("""
### About
This application uses a deep learning model to classify chest X-ray images into two categories:
- Normal
- Adenocarcinoma Cancer

Please note that this is an AI-assisted tool and should not be used as a sole diagnostic tool. Always consult with healthcare professionals for medical decisions.
""")