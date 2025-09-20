# Master

Script con preprocesamiento de descripciones



## Running the App

### Local Development
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
streamlit run dashboard.py
```

The app will automatically install the Spanish spacy model during deployment.

### Resources
- Main dashboard: `dashboard.py`
- Description preprocessing: `preprocess_descriptions.py`
- Requirements: `requirements.txt`
