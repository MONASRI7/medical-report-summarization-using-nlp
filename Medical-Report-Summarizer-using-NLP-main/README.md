# Medical Report Summarizer

A powerful web application that uses Natural Language Processing (NLP) to analyze and summarize medical reports. The application can process various types of medical documents, extract key information, and generate concise summaries.

## Features

- **Text Summarization**: Automatically generates concise summaries of medical reports using TF-IDF and sentence scoring
- **Key Information Extraction**: Identifies and categorizes important medical information:
  - Diagnoses
  - Medications
  - Procedures
  - Symptoms
  - Tests
  - Recommendations
- **File Support**: Process multiple file formats:
  - Text files (.txt)
  - PDF documents (.pdf)
  - Word documents (.doc, .docx)
- **History Management**: 
  - Saves all processed reports
  - View past summaries
  - Export summaries in multiple formats
- **Export Options**:
  - PDF format
  - Word document
  - CSV format
- **Modern UI**:
  - Responsive design
  - Real-time processing
  - Interactive history view
  - Mobile-friendly interface

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Medical-Report-Summarizer-using-NLP.git
cd Medical-Report-Summarizer-using-NLP
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the application:
   - Enter medical text directly in the text area
   - Upload medical reports in supported formats
   - View generated summaries and key information
   - Access history of processed reports
   - Export summaries in your preferred format

## Project Structure

```
Medical-Report-Summarizer-using-NLP/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── instance/             # Database directory
│   └── history.db        # SQLite database
├── static/               # Static files
│   ├── css/             # Stylesheets
│   └── js/              # JavaScript files
├── templates/            # HTML templates
│   ├── index.html       # Main application page
│   ├── history.html     # History view page
│   └── landing.html     # Landing page
└── uploads/             # Temporary file storage
```

## Technologies Used

- **Backend**:
  - Flask (Python web framework)
  - NLTK (Natural Language Processing)
  - scikit-learn (Machine Learning)
  - SQLite (Database)
  - PyPDF2 (PDF processing)
  - python-docx (Word document processing)

- **Frontend**:
  - HTML5
  - Tailwind CSS
  - JavaScript
  - Font Awesome icons

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NLTK library for NLP capabilities
- Flask framework for web application
- Tailwind CSS for modern UI design
- All contributors who have helped improve this project