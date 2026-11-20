import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Biomedical Entity Extractor", page_icon="??", layout="wide")
MODEL_ID = "d4data/biomedical-ner-all"

@st.cache_resource
def load_ner_pipeline():
    return pipeline("token-classification", model=MODEL_ID, aggregation_strategy="simple")

st.title("?? Biomedical Named Entity Recognition ^(NER^)")
st.write("Paste a clinical note or biomedical abstract below to extract entities.")

sample_text = "A 54-year-old patient presented with acute dyspnea, chest tightness, and palpitations. Echocardiography showed moderate mitral stenosis, and the patient was started on metoprolol 25 mg twice daily."
text_input = st.text_area("Clinical / Biomedical Text", value=sample_text, height=150)

if st.button("Extract Entities", type="primary"):
    with st.spinner("Loading model and extracting entities..."):
        ner = load_ner_pipeline()
        entities = ner(text_input)
    if not entities:
        st.info("No biomedical entities detected.")
    else:
        st.dataframe([{"Entity": e["word"], "Label": e["entity_group"], "Confidence": round(float(e["score"]), 4)} for e in entities], use_container_width=True)
