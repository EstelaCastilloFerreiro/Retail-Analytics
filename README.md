# Trucco_clean

Script de Trucco limpio y con preprocesamiento de descripciones

## About

Script de Trucco limpio y con preprocesamiento de descripciones

## Running the App

### Local Development
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
streamlit run dashboard.py
```

### Streamlit Cloud Deployment
1. Go to https://share.streamlit.io/
2. Sign in with your GitHub account
3. Select this repository: `carlotapprieto/Trucco_clean`
4. Set the main file path to: `dashboard.py`
5. Deploy!

The app will automatically install the Spanish spacy model during deployment.

### Resources
- Main dashboard: `dashboard.py`
- Description preprocessing: `preprocess_descriptions.py`
- Requirements: `requirements.txt`
