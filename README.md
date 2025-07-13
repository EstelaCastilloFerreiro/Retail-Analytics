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

### GitHub Codespaces
1. Open this repository in GitHub Codespaces
2. In the terminal, run:
   ```bash
   pip install -r requirements.txt
   python -m spacy download es_core_news_sm
   streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
   ```
3. Click "Open in Browser" when prompted, or manually open the forwarded port

### Resources
- Main dashboard: `dashboard.py`
- Description preprocessing: `preprocess_descriptions.py`
- Requirements: `requirements.txt`
