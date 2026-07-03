from flask import Flask, render_template, request, send_file
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Skills Database
SKILLS = [
    "python",
    "sql",
    "flask",
    "machine learning",
    "deep learning",
    "java",
    "git",
    "github",
    "data analysis",
    "excel",
    "power bi",
    "tableau",
    "numpy",
    "pandas",
    "opencv",
    "tensorflow",
    "pytorch",
    "artificial intelligence",
    "data visualization",
    "statistics",
    "problem solving",
    "mysql",
    "postgresql",
    "scikit-learn"
]

# Global variable for PDF report
latest_report = {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    global latest_report

    file = request.files["resume"]

    if file.filename == "":
        return "Please upload a PDF file."

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Extract Resume Text
    reader = PdfReader(filepath)

    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text

    # Job Description
    job_description = request.form["job_description"]

    # TF-IDF Similarity
    documents = [
        resume_text,
        job_description
    ]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    match_score = round(
        similarity[0][0] * 100,
        2
    )

    # Skill Matching
    resume_lower = resume_text.lower()
    jd_lower = job_description.lower()

    matched_skills = []
    missing_skills = []

    for skill in SKILLS:

        if skill in jd_lower:

            if skill in resume_lower:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)

    # Recommendations
    suggestions = []

    for skill in missing_skills:
        suggestions.append(
            f"Consider learning {skill} and adding related projects."
        )

    # Skills Match Percentage
    total_skills = len(matched_skills) + len(missing_skills)

    if total_skills > 0:
        skills_match_percentage = round(
            (len(matched_skills) / total_skills) * 100,
            2
        )
    else:
        skills_match_percentage = 0

    # SMART ATS SCORE

    skills_points = min(
        len(matched_skills) * 5,
        30
    )

    project_points = (
        20 if "project" in resume_lower else 0
    )

    certificate_points = (
        15 if (
            "certification" in resume_lower
            or "certificate" in resume_lower
        ) else 0
    )

    github_points = (
        10 if "github" in resume_lower else 0
    )

    linkedin_points = (
        10 if "linkedin" in resume_lower else 0
    )

    experience_points = (
        10 if "experience" in resume_lower else 0
    )

    education_points = (
        5 if (
            "education" in resume_lower
            or "bachelor" in resume_lower
            or "b.tech" in resume_lower
        ) else 0
    )

    ats_score = (
        skills_points
        + project_points
        + certificate_points
        + github_points
        + linkedin_points
        + experience_points
        + education_points
    )

    ats_score = min(ats_score, 100)

    # Resume Category Detection

    resume_category = "General Profile"

    if (
        "machine learning" in resume_lower
        or "tensorflow" in resume_lower
        or "artificial intelligence" in resume_lower
    ):
        resume_category = "AI / ML Engineer"

    elif (
        "data analysis" in resume_lower
        or "power bi" in resume_lower
        or "tableau" in resume_lower
    ):
        resume_category = "Data Analyst"

    elif (
        "flask" in resume_lower
        or "python" in resume_lower
    ):
        resume_category = "Python Developer"

    # Store data for PDF report

    latest_report = {
        "ats_score": ats_score,
        "match_score": match_score,
        "resume_category": resume_category,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,

        "skills_points": skills_points,
        "project_points": project_points,
        "certificate_points": certificate_points,
        "github_points": github_points,
        "linkedin_points": linkedin_points,
        "experience_points": experience_points,
        "education_points": education_points
    }

    return render_template(
        "result.html",
        resume_text=resume_text,
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        suggestions=suggestions,
        skills_match_percentage=skills_match_percentage,
        ats_score=ats_score,
        resume_category=resume_category,

        matched_count=len(matched_skills),
        missing_count=len(missing_skills),

        skills_points=skills_points,
        project_points=project_points,
        certificate_points=certificate_points,
        github_points=github_points,
        linkedin_points=linkedin_points,
        experience_points=experience_points,
        education_points=education_points
    )


@app.route("/download-report")
def download_report():

    global latest_report

    pdf_file = "TalentSync_Report.pdf"

    c = canvas.Canvas(pdf_file)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(180, 800, "TalentSync AI Report")

    c.setFont("Helvetica", 12)

    c.drawString(
        50,
        760,
        f"Resume Category: {latest_report.get('resume_category', '')}"
    )

    c.drawString(
        50,
        735,
        f"ATS Score: {latest_report.get('ats_score', 0)}/100"
    )

    c.drawString(
        50,
        710,
        f"Match Score: {latest_report.get('match_score', 0)}%"
    )

    y = 670

    c.drawString(50, y, "Matched Skills:")
    y -= 20

    for skill in latest_report.get("matched_skills", []):
        c.drawString(70, y, f"- {skill}")
        y -= 20

    y -= 10

    c.drawString(50, y, "Missing Skills:")
    y -= 20

    for skill in latest_report.get("missing_skills", []):
        c.drawString(70, y, f"- {skill}")
        y -= 20

    y -= 10

    c.drawString(50, y, "Recommendations:")
    y -= 20

    for suggestion in latest_report.get("suggestions", []):
        c.drawString(
            70,
            y,
            f"- {suggestion[:70]}"
        )
        y -= 20

    c.save()

    return send_file(
        pdf_file,
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)