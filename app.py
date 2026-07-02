from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["resume"]

    if file.filename == "":
        return "Please upload a PDF file."

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    # Extract PDF Text
    reader = PdfReader(filepath)

    resume_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            resume_text += text

    # Get Job Description
    job_description = request.form["job_description"]

    # TF-IDF Matching
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
            f"Consider learning {skill} and adding related projects to your resume."
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

    return render_template(
        "result.html",
        resume_text=resume_text,
        match_score=match_score,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        suggestions=suggestions,
        skills_match_percentage=skills_match_percentage
    )


if __name__ == "__main__":
    app.run(debug=True)