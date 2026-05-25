# ResumeIQ — AI-Powered Resume Analyzer

A Flask web application that analyzes your resume against a job description using
TF-IDF / cosine similarity and a built-in skills database to give you a match score,
gap analysis, and actionable suggestions.

---

## 📁 Project Structure

```
resume_analyzer/
├── app.py               # Main Flask backend (routes + NLP logic)
├── requirements.txt     # Python dependencies
├── uploads/             # Uploaded PDFs (auto-created on first run)
├── templates/
│   ├── home.html        # Landing page
│   ├── upload.html      # Upload form
│   └── results.html     # Analysis results
└── static/
    └── css/
        └── style.css    # All styling
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## 🔧 How It Works

| Step | What Happens |
|------|-------------|
| **Upload** | User uploads a PDF resume; `pdfplumber` extracts all text |
| **Input** | User pastes a job description into the text area |
| **TF-IDF** | Both texts are vectorized using TF-IDF with bigrams |
| **Cosine Similarity** | The two vectors are compared; result × 100 = match % |
| **Skill Extraction** | Regex matching against a 100+ entry skills database |
| **Gap Analysis** | Set difference: job_skills − resume_skills = missing |
| **Suggestions** | Rule-based recommendations based on score + gaps |

---

## 🌐 API Endpoint

POST `/api/analyze`  
Content-Type: `multipart/form-data`  
Fields: `resume` (PDF file), `job_description` (string)

Returns JSON:
```json
{
  "match_score": 72.4,
  "score_label": "Good Match",
  "resume_skills": ["python", "flask", "sql"],
  "job_skills": ["python", "flask", "docker", "kubernetes"],
  "matching_skills": ["python", "flask"],
  "missing_skills": ["docker", "kubernetes"]
}
```

---

## 📦 Dependencies

- **Flask** — web framework
- **pdfplumber** — PDF text extraction
- **scikit-learn** — TF-IDF vectorization + cosine similarity
- **Werkzeug** — secure file uploads

---

## 🚀 Optional Enhancements

- Add **spaCy** for named-entity-based skill extraction
- Use **sqlite3** to store past analyses
- Deploy to **Railway / Render / Fly.io** with a `Procfile`
- Add **authentication** with Flask-Login

---

## 📄 License

MIT — use freely, credit appreciated.
