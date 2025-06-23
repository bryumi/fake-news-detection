
# Comparação de Modelos: Fake News (BERT vs. LSTM vs. Random Forest)

## Pré-requisitos
- Python 3.8+
- VSCode
- Virtualenv (opcional)

## Instalação
1. Crie um ambiente virtual (opcional):
   python -m venv venv
   .\venv\Scripts\activate

2. Instale as dependências:
   pip install -r requirements.txt

3. Coloque o dataset baixado da Kaggle na pasta `data/` com o nome: fake_or_real_news.csv

4. Execute o script:
   python main.py

## Observação
O modelo BERT usa um exemplo simplificado de pipeline da Hugging Face. Para treinar o BERT completo, é necessário um ajuste fino (fine-tuning), não incluso aqui por simplicidade.
