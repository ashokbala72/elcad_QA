import streamlit as st
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import io
from processor.layoutlm_extractor import extract_layout_text
from processor.symbol_classifier import classify_symbols
from processor.rule_checker import check_rules

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(layout="wide")
st.title("AI-Powered ELCAD Drawing QA Assistant")

uploaded_file = st.file_uploader("📥 Upload ELCAD PDF Drawing", type=["pdf"])
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    st.success(f"✅ Loaded {len(doc)} pages")

    for page_num in range(len(doc)):
        st.subheader(f"📄 Page {page_num + 1}")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

        with st.spinner("🔍 Running AI QA..."):
            layout_output = extract_layout_text(image)
            symbol_output = classify_symbols(image)
            report = check_rules(layout_output, symbol_output, OPENAI_API_KEY)

        # Visual overlay of symbols
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)

        for sym in symbol_output:
            x, y = sym["position"]
            draw.rectangle([x, y, x+60, y+60], outline="red", width=2)
            draw.text((x+3, y+3), sym["label"], fill="red")

        st.image(annotated_image, caption=f"🖼️ Annotated Page {page_num + 1}", use_column_width=True)

        st.subheader("🔎 Detected Issues")
        st.json(report)
