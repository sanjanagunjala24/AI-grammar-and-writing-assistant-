import streamlit as st
from ollama import chat
from pypdf import PdfReader
from faster_whisper import WhisperModel

st.set_page_config(page_title="AI Writing Assistant")

st.title("✍️ AI Grammar & Writing Assistant")

# Ollama
def ask_ai(text):
    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": f"Correct the grammar and improve this text:\n\n{text}"
            }
        ]
    )
    return response["message"]["content"]


# Text input
text = st.text_area("Enter your text")

if st.button("Correct Grammar"):
    if text:
        with st.spinner("Correcting..."):
            result = ask_ai(text)
        st.subheader("Result")
        st.write(result)


# PDF upload
pdf = st.file_uploader("Upload PDF", type=["pdf"])

if pdf:
    reader = PdfReader(pdf)
    pdf_text = "\n".join(
        page.extract_text() or ""
        for page in reader.pages
    )

    st.subheader("PDF Text")
    st.write(pdf_text[:5000])


# Audio transcription
audio = st.file_uploader(
    "Upload Audio",
    type=["mp3", "wav", "m4a"]
)

if audio and st.button("Transcribe Audio"):

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

    segments, _ = model.transcribe(audio)

    transcript = " ".join(
        segment.text for segment in segments
    )

    st.subheader("Transcript")
    st.write(transcript)