import streamlit as st
import fitz  # PyMuPDF
from fpdf import FPDF
import tempfile
import unicodedata

# Utility to clean Unicode text safely
def safe_text(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

# PDF Generator
def generate_combined_pdf_report(project_number, contact_person, page_reports):
    class ReportPDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, "Guidelines Monitoring Report", ln=True, align="C")
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    pdf = ReportPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)

    # Cover Page
    pdf.add_page()
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Project Number: {safe_text(project_number)}", ln=True)
    pdf.cell(0, 10, f"Contact Person: {safe_text(contact_person)}", ln=True)
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Page-by-Page Findings Summary:", ln=True)

    # Add each page's issues
    for page_num, issues in page_reports:
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Findings for Page {page_num + 1}:", ln=True)
        pdf.set_font("Arial", "", 11)
        for issue in issues:
            pdf.multi_cell(0, 8, f"- {safe_text(issue)}")
        pdf.ln(5)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        pdf.output(tmp_file.name)
        return tmp_file.name

# Streamlit App Start
st.set_page_config(layout="wide")
st.title("📐 AI-Powered ELCAD Drawing QA Assistant")

uploaded_file = st.file_uploader("📥 Upload ELCAD PDF Drawing", type="pdf")

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    total_pages = len(doc)
    st.success(f"✅ Loaded {total_pages} pages")

    page_wise_issues = []

    for i, page in enumerate(doc):
        st.subheader(f"📄 Page {i + 1}")
        image = page.get_pixmap(dpi=150).tobytes("png")
        st.image(image, caption=f"Page {i + 1}", use_column_width=True)

        # Dummy issues (replace with actual model outputs)
        issues = [
            f"1. Missing Labels: Relay symbol on page {i+1} lacks a nearby label.",
            f"2. Misclassified Symbol: 'Cable' classification may be incorrect.",
            f"3. Zone B3 mentioned but no matching symbol found.",
            f"4. Detected symbol at (150,0) far from expected layout zone.",
            f"5. Naming mismatch: '38', 'CT1', and 'K1' have no linked symbols.",
            f"6. Inconsistent confidence values for symbols."
        ]

        st.markdown("🔎 **Detected Issues**")
        st.json({"issues": issues})
        page_wise_issues.append((i, issues))

        # Only on the last page
        if i == total_pages - 1:
            st.markdown("---")
            st.subheader("🚨 Repeated Error Detected Across Pages")
            repeated = issues[0]  # Simulate the most common issue
            st.markdown(f"**Most Frequent Issue:** {repeated}")
            st.button("📧 Send Email to Contractor (Simulated)")

            # Generate and offer PDF download
            combined_pdf_path = generate_combined_pdf_report(
                project_number="100034567",
                contact_person="Ashok Balasubramanian",
                page_reports=page_wise_issues
            )

            with open(combined_pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download Combined Report (PDF)",
                    data=f.read(),
                    file_name="elcad_combined_qa_report.pdf",
                    mime="application/pdf"
                )
