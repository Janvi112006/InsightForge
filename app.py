
import os
import fitz
from flask import Flask, render_template, request
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@app.route("/", methods=["GET", "POST"])
def home():
    summary = None
    error = None

    if request.method == "POST":
        file = request.files.get("pdf")

        if not file or not file.filename.lower().endswith(".pdf"):
            error = "Please upload a valid PDF file."
        else:
            try:
                pdf = fitz.open(
                    stream=file.read(),
                    filetype="pdf"
                )

                paper_text = ""
                for page_number, page in enumerate(pdf, start=1):
                    paper_text += (
                        f"\n[Page {page_number}]\n"
                        + page.get_text()
                    )

                if not paper_text.strip():
                    raise ValueError("No readable text found in this PDF.")

                paper_text = paper_text[:30000]

                # Stage 1: Extract information from the paper
                extraction_prompt = f"""
                Read the research paper carefully.
                Extract only information explicitly stated in it.

                Identify:
                1. Research problem
                2. Objectives
                3. Methodology
                4. Dataset
                5. Key results
                6. Limitations

                Mention page numbers where possible.
                Write "Not stated" if information is missing.
                Do not invent information.

                PAPER:
                {paper_text}
                """

                stage1 = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=extraction_prompt
                )

                # Stage 2: Create and check the final summary
                summary_prompt = f"""
                Create a simple, student-friendly research paper summary
                using the extracted information below.

                Use these headings:
                Research Problem
                Objectives
                Methodology
                Dataset
                Key Results
                Limitations

                Check that every statement is supported by the
                extracted information. Do not add outside facts.
                Mark missing information as "Not stated".

                EXTRACTED INFORMATION:
                {stage1.text}
                """

                stage2 = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=summary_prompt
                )

                summary = stage2.text

            except Exception as e:
                error = f"Something went wrong: {str(e)}"

    return render_template(
        "index.html",
        summary=summary,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
