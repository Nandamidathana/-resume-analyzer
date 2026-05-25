"""
Resume Analyzer - Flask Backend
Analyzes resume against job description using NLP techniques
"""

import os
import re
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── App Configuration ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max file size
app.config['SECRET_KEY'] = 'resume-analyzer-secret-2024'

ALLOWED_EXTENSIONS = {'pdf'}

# ─── Comprehensive Skills Database ────────────────────────────────────────────
SKILLS_DATABASE = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "rust", "php", "swift", "kotlin", "scala", "r", "matlab", "perl",

    # Web Technologies
    "html", "css", "react", "angular", "vue", "nodejs", "node.js", "express",
    "django", "flask", "fastapi", "spring", "laravel", "rails", "next.js",
    "nuxt", "graphql", "rest", "restful", "api", "bootstrap", "tailwind",
    "sass", "jquery",

    # Data & ML
    "machine learning", "deep learning", "neural networks", "tensorflow",
    "pytorch", "keras", "scikit-learn", "pandas", "numpy", "matplotlib",
    "data analysis", "data science", "nlp", "computer vision", "ai",
    "artificial intelligence", "statistics", "tableau", "power bi",
    "ensemble", "random forest", "data structures",

    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "cassandra", "dynamodb", "firebase", "neo4j",

    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "ci/cd",
    "terraform", "ansible", "linux", "git", "github", "gitlab", "bitbucket",
    "microservices", "serverless", "devops", "sre",

    # Soft Skills
    "leadership", "communication", "teamwork", "problem solving",
    "problem-solving", "agile", "scrum", "project management",
    "collaboration", "analytical", "critical thinking",

    # Security
    "cybersecurity", "penetration testing", "ethical hacking", "siem",
    "firewalls", "encryption", "oauth", "jwt", "ssl/tls",

    # Mobile
    "android", "ios", "react native", "flutter", "xamarin",

    # Tools & Others
    "jira", "confluence", "slack", "figma", "photoshop", "excel",
    "word", "powerpoint", "selenium", "jest", "pytest", "junit",
    "algorithms", "leetcode", "competitive programming",
}


# ─── Helper Functions ──────────────────────────────────────────────────────────

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(filepath):
    """Extract all text from a PDF using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return ""
    return text.strip()


def clean_text(text):
    """Clean and normalize text for analysis."""
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text):
    """
    Extract skills from text by matching against the skills database.
    Handles both single-word and multi-word skills.
    """
    text_lower = clean_text(text)
    found_skills = set()

    for skill in SKILLS_DATABASE:
        # Use word boundary matching for accurate extraction
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)

    return sorted(found_skills)


def calculate_tfidf_score(resume_text, job_text):
    """
    Pure TF-IDF cosine similarity between resume and job description.
    Works best when both texts are similar in length.
    """
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return float(similarity[0][0]) * 100
    except Exception:
        return 0


def calculate_skill_overlap_score(resume_skills, job_skills):
    """
    Skill-based score: what % of job-required skills appear in the resume.
    Formula: (matching skills / total job skills) * 100
    This is immune to document length differences.
    """
    if not job_skills:
        return 0
    matching = set(resume_skills) & set(job_skills)
    return (len(matching) / len(job_skills)) * 100


def calculate_keyword_density_score(resume_text, job_text):
    """
    Measures how many unique important words from the job description
    appear anywhere in the resume. Handles short resume text fairly.
    """
    job_words = set(re.findall(r'\b[a-z]{4,}\b', job_text.lower()))
    resume_words = set(re.findall(r'\b[a-z]{4,}\b', resume_text.lower()))

    # Remove common stop words manually
    stop = {'with', 'this', 'that', 'from', 'have', 'will', 'your', 'they',
            'been', 'also', 'into', 'more', 'some', 'than', 'their', 'there',
            'when', 'what', 'work', 'using', 'about', 'should', 'experience'}
    job_words -= stop

    if not job_words:
        return 0
    overlap = job_words & resume_words
    return (len(overlap) / len(job_words)) * 100


def calculate_match_score(resume_text, job_text):
    """
    HYBRID scoring — combines three signals to give a fair score
    even for short resumes:

      • Skill overlap  (50%) — most important: did you list the right skills?
      • Keyword density(30%) — do your resume words appear in the JD?
      • TF-IDF cosine  (20%) — overall text similarity (penalised less now)

    Returns a score between 0 and 100.
    """
    if not resume_text.strip() or not job_text.strip():
        return 0

    cleaned_resume = clean_text(resume_text)
    cleaned_job    = clean_text(job_text)

    resume_skills = extract_skills(resume_text)
    job_skills    = extract_skills(job_text)

    # Three component scores
    skill_score   = calculate_skill_overlap_score(resume_skills, job_skills)
    keyword_score = calculate_keyword_density_score(cleaned_resume, cleaned_job)
    tfidf_score   = calculate_tfidf_score(cleaned_resume, cleaned_job)

    # Weighted combination  (weights sum to 1.0)
    hybrid = (skill_score * 0.50) + (keyword_score * 0.30) + (tfidf_score * 0.20)

    # Print breakdown for debugging
    print(f"[Score Breakdown] Skill: {skill_score:.1f}% | "
          f"Keyword: {keyword_score:.1f}% | TF-IDF: {tfidf_score:.1f}% | "
          f"Final: {hybrid:.1f}%")

    return round(min(hybrid, 100), 1)


def get_score_label(score):
    """Return a descriptive label based on match score."""
    if score >= 80:
        return ("Excellent Match", "excellent", "🎉")
    elif score >= 60:
        return ("Good Match", "good", "👍")
    elif score >= 40:
        return ("Moderate Match", "moderate", "📊")
    else:
        return ("Low Match", "low", "⚠️")


def generate_suggestions(missing_skills, score, resume_skills, job_skills):
    """Generate actionable improvement suggestions based on analysis."""
    suggestions = []

    # Skill-based suggestions
    if missing_skills:
        top_missing = missing_skills[:5]
        suggestions.append({
            "type": "skills",
            "icon": "🎯",
            "title": "Add Missing Skills",
            "detail": f"Include these key skills in your resume: {', '.join(top_missing)}"
        })

    # Score-based suggestions
    if score < 40:
        suggestions.append({
            "type": "keywords",
            "icon": "🔑",
            "title": "Improve Keyword Density",
            "detail": "Your resume lacks many keywords from the job description. Mirror the exact language used in the job posting."
        })
    elif score < 70:
        suggestions.append({
            "type": "keywords",
            "icon": "🔑",
            "title": "Optimize Keyword Usage",
            "detail": "Add more job-specific keywords naturally throughout your experience bullets and summary section."
        })

    # Experience suggestions
    suggestions.append({
        "type": "format",
        "icon": "📝",
        "title": "Use Action Verbs",
        "detail": "Start each bullet point with strong action verbs like 'Developed', 'Led', 'Implemented', 'Optimized', 'Delivered'."
    })

    # Quantify achievements
    suggestions.append({
        "type": "impact",
        "icon": "📈",
        "title": "Quantify Achievements",
        "detail": "Add numbers and metrics to show impact. E.g., 'Improved performance by 40%' or 'Managed a team of 8 engineers'."
    })

    # ATS optimization
    suggestions.append({
        "type": "ats",
        "icon": "🤖",
        "title": "ATS Optimization",
        "detail": "Use a clean, single-column format. Avoid tables and graphics. Use standard section headings like 'Experience', 'Education', 'Skills'."
    })

    # If good match
    if score >= 70:
        suggestions.append({
            "type": "polish",
            "icon": "✨",
            "title": "Polish Your Summary",
            "detail": "Great match! Tailor your professional summary to directly echo the job's top 2-3 requirements."
        })

    return suggestions


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    """Home / Landing page."""
    return render_template('home.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    GET:  Show the upload form.
    POST: Process the uploaded resume + job description.
    """
    if request.method == 'GET':
        return render_template('upload.html')

    # ── Validate Inputs ──────────────────────────────────────────────────────
    if 'resume' not in request.files:
        return render_template('upload.html', error="Please upload a resume PDF file.")

    file = request.files['resume']
    job_description = request.form.get('job_description', '').strip()

    if file.filename == '':
        return render_template('upload.html', error="No file selected.")

    if not allowed_file(file.filename):
        return render_template('upload.html', error="Only PDF files are supported.")

    if not job_description:
        return render_template('upload.html', error="Please paste a job description.")

    if len(job_description) < 50:
        return render_template('upload.html', error="Job description is too short. Please provide a complete job description.")

    # ── Save & Extract Resume ─────────────────────────────────────────────────
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)

    if not resume_text:
        return render_template('upload.html', error="Could not extract text from PDF. Please ensure it's a text-based PDF (not scanned).")

    # ── Perform Analysis ──────────────────────────────────────────────────────
    score = calculate_match_score(resume_text, job_description)
    score_label, score_class, score_emoji = get_score_label(score)

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)

    # Find missing skills (in job but not in resume)
    missing_skills = sorted(set(job_skills) - set(resume_skills))

    # Find matching skills (in both)
    matching_skills = sorted(set(resume_skills) & set(job_skills))

    # Generate suggestions
    suggestions = generate_suggestions(missing_skills, score, resume_skills, job_skills)

    # Word count stats
    resume_word_count = len(resume_text.split())
    job_word_count = len(job_description.split())

    # Build results dict
    results = {
        "score": score,
        "score_label": score_label,
        "score_class": score_class,
        "score_emoji": score_emoji,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "suggestions": suggestions,
        "resume_name": filename,
        "resume_word_count": resume_word_count,
        "job_word_count": job_word_count,
    }

    return render_template('results.html', **results)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    JSON API endpoint for programmatic access.
    Accepts multipart/form-data with 'resume' file and 'job_description' text.
    """
    if 'resume' not in request.files:
        return jsonify({"error": "No resume file provided"}), 400

    file = request.files['resume']
    job_description = request.form.get('job_description', '')

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files allowed"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    resume_text = extract_text_from_pdf(filepath)
    score = calculate_match_score(resume_text, job_description)
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    missing_skills = sorted(set(job_skills) - set(resume_skills))
    matching_skills = sorted(set(resume_skills) & set(job_skills))

    return jsonify({
        "match_score": score,
        "score_label": get_score_label(score)[0],
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills
    })


# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
