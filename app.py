import streamlit as st
import os
from groq import Groq
import json
import time

# Setup page config
st.set_page_config(
    page_title="Medical SOAP Note Generator",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

def initialize_groq():
    """Initialize Groq client with API key"""
    api_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = api_key
    return Groq()

def generate_soap_note(client, symptoms, examination, additional_info=""):
    """Generate SOAP note using Groq API"""
    system_message = """You are a medical professional assistant helping to generate SOAP notes. 
    Create a detailed, professional SOAP note based on the provided information.
    Format the note with clear sections for Subjective, Objective, Assessment, and Plan.
    Use proper medical terminology while maintaining clarity."""
    
    prompt = f"""
    Create a detailed SOAP note based on the following information:
    
    Patient Symptoms: {symptoms}
    Physical Examination: {examination}
    Additional Information: {additional_info}
    
    Please format the note professionally with clear sections for Subjective, Objective, Assessment, and Plan.
    """
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-llama-70b",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_completion_tokens=4096,
            top_p=0.95,
            stream=True,
            stop=None
        )
        
        # Initialize response string
        response = ""
        
        # Create a placeholder for streaming output
        note_placeholder = st.empty()
        
        # Stream the response
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            response += content
            note_placeholder.markdown(response)
            time.sleep(0.01)
        
        return response
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None

def save_note_to_history(note, symptoms, examination):
    """Save generated note to history"""
    st.session_state.history.append({
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'symptoms': symptoms,
        'examination': examination,
        'note': note
    })

def main():
    st.title("🏥 Medical SOAP Note Generator")
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Generate Note", "History"])
    
    with tab1:
        st.markdown("### Patient Information")
        
        # Input fields
        symptoms = st.text_area("Patient Symptoms", 
            height=100,
            placeholder="Enter patient's symptoms and complaints...")
            
        examination = st.text_area("Physical Examination", 
            height=100,
            placeholder="Enter physical examination findings...")
            
        additional_info = st.text_area("Additional Information (optional)",
            height=50,
            placeholder="Enter any additional relevant information...")
        
        # Generate button
        if st.button("Generate SOAP Note", type="primary"):
            if symptoms and examination:
                with st.spinner("Generating SOAP note..."):
                    client = initialize_groq()
                    note = generate_soap_note(client, symptoms, examination, additional_info)
                    
                    if note:
                        save_note_to_history(note, symptoms, examination)
                        
                        # Add download button
                        st.download_button(
                            label="Download SOAP Note",
                            data=note,
                            file_name=f"soap_note_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            else:
                st.warning("Please enter both symptoms and examination findings.")
    
    with tab2:
        st.markdown("### Previously Generated Notes")
        if st.session_state.history:
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Note {len(st.session_state.history)-i}: {entry['timestamp']}"):
                    st.markdown("**Symptoms:**")
                    st.text(entry['symptoms'])
                    st.markdown("**Examination:**")
                    st.text(entry['examination'])
                    st.markdown("**Generated Note:**")
                    st.markdown(entry['note'])
                    
                    # Download button for historical notes
                    st.download_button(
                        label="Download This Note",
                        data=entry['note'],
                        file_name=f"soap_note_{entry['timestamp'].replace(' ', '_')}.txt",
                        mime="text/plain",
                        key=f"download_{i}"
                    )
        else:
            st.info("No previous notes found.")

if __name__ == "__main__":
    main()