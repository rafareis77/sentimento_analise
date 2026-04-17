# 🤖 Análise de Sentimentos com Machine Learning

Projeto completo de classificação de sentimentos em reviews de produtos, desenvolvido em Python com Jupyter Notebook.  
Abrange todo o ciclo de vida de um modelo de Machine Learning: da análise exploratória ao deploy, passando por limpeza de dados, engenharia de atributos, treinamento, otimização e salvamento do modelo.

---

## 📁 Estrutura do Projeto

```
Analise-Sentimentos/
├── Analise-Sentimentos.ipynb   
├── Modelo.ipynb                
├── dataset.csv                 
├── modelo_sentimentov1.joblib  
└── requirements.txt            
```

---

## ⚙️ O que o projeto faz

### 1. Carregamento e EDA
Leitura do `dataset.csv` com Pandas e análise exploratória inicial: shape, valores ausentes e distribuição das classes de sentimento (positivo/negativo).

### 2. Limpeza de Dados
- Remoção de valores ausentes
- Normalização de texto com função `limpa_texto`: converte para minúsculas, remove acentos, pontuações, números e espaços extras

### 3. Engenharia de Atributos
- Criação da coluna `texto_limpo`
- Mapeamento dos sentimentos para valores numéricos (`positivo` → 1, `negativo` → 0)

### 4. Pipeline de Modelagem
Pipeline com três etapas sequenciais:
- **TF-IDF** — vetorização do texto
- **StandardScaler** — padronização dos vetores
- **Regressão Logística** — classificação final

### 5. Otimização de Hiperparâmetros
Uso de `GridSearchCV` com validação cruzada de 5 folds para encontrar a melhor combinação de parâmetros do pipeline.

### 6. Avaliação do Modelo
- Acurácia
- Relatório de classificação (precisão, recall, F1-score)
- Matriz de confusão

### 7. Deploy e Uso do Modelo
- Salvamento do modelo treinado com `joblib` (`modelo_sentimentov1.joblib`)
- Carregamento e uso em produção via `Modelo.ipynb`
- Classificação de novos reviews com a função `prever_sentimentos`

---

## 🚀 Como executar

### Pré-requisitos

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Treinando o modelo

```bash
jupyter notebook Analise-Sentimentos.ipynb
```

### Usando o modelo em produção

```bash
jupyter notebook Modelo.ipynb
```

---

## 🛠️ Tecnologias utilizadas

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
