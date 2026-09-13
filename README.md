# HealthSearch

Protótipo Streamlit em um único arquivo que pesquisa exatamente os seis documentos da atividade. Combina BM25 e busca semântica com RRF. Sem Cross-Encoder.

## Instalação e execução

Use Python 3.10 ou superior. Na pasta do projeto:

```bash
python -m venv .venv
```

Ative no Windows (PowerShell): `.venv\Scripts\Activate.ps1`; no Linux/macOS: `source .venv/bin/activate`.

```bash
python -m pip install -r requirements.txt
python -m streamlit run healthsearch_app.py
```

Bibliotecas: Streamlit (interface/cache), pandas (tabelas), NumPy (vetores/cosseno), rank_bm25 (BM25) e sentence-transformers (embeddings).

## Como funciona

- **BM25:** pesquisa título e conteúdo, com minúsculas, remoção de acentos/pontuação e stopwords básicas. Preserva hífens internos: `CÓD-ECG-12D` vira `cod-ecg-12d`. `k1` controla saturação de frequência e `b`, normalização pelo tamanho. Em `k1=0`, calcula o limite por presença do termo para evitar divisão por zero da biblioteca.
- **Semântico:** usa `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` em CPU e similaridade de cosseno. Modelo, embeddings do corpus e resultados por consulta têm cache. Títulos e conteúdos originais são enviados ao modelo, sem a limpeza léxica.
- **RRF:** `alpha / (60 + posição_BM25) + (1 - alpha) / (60 + posição_semântica)`. Posições começam em 1; scores são ordenados do maior para o menor, com desempate pelo ID. `alpha=1` prioriza totalmente BM25; `alpha=0`, semântico.

As quatro abas mostram rankings e uma comparação de posições/scores. Os sliders recalculam BM25/RRF. Consulta vazia, pontuação ou apenas stopwords exibem uma orientação.

## Modelo e alternativa permitida

A execução é local, sem API paga, chave ou serviço remoto de inferência. A instalação e o primeiro download do modelo exigem internet; depois de armazenado em cache, o modelo pode funcionar offline.

Se a importação, o carregamento ou a execução do modelo falhar, a aplicação ativa **simulação vetorial documentada**, com aviso e motivo na tela: oito dimensões binárias de conceitos médicos (infarto, medicamentos, hipertensão, AVC, reanimação, ECG, monitorização e dor torácica), reconhecidos por termos/sinônimos manuais, e cosseno. Não são embeddings aprendidos; termos fora desses grupos podem resultar em vetor zero. Reinicie o Streamlit após corrigir o modelo para limpar resultados simulados do cache.

## Consultas para experimentar

- Termo/código exato: `CÓD-ECG-12D`, `telemetria`, `trombolíticos`.
- Sinônimos: `ataque cardíaco` / `infarto`, `derrame` / `acidente vascular cerebral`, `pressão alta` / `hipertensão`, `aspirina` / `ácido acetilsalicílico`, `dor no peito` / `dor precordial`.

Compare os rankings sem pressupor que o híbrido sempre será melhor. A base é pequena, não há avaliação quantitativa de relevância e RRF gera scores positivos mesmo para documentos sem correspondência. Scores não são probabilidades.

**Escopo da entrega:** os quatro arquivos solicitados. O enunciado também exige relatório técnico PDF de até duas páginas, gráfico de ranks e divisão de tarefas da equipe; esse relatório permanece pendente e o TXT não o substitui.

## Verificação realizada

Compilação/importação com Python 3.10.10; conferência dos seis documentos contra o DOCX; testes de BM25 (incluindo k1=0 e efeitos dos parâmetros), cosseno, RRF e ordenação. O Streamlit AppTest verificou as quatro abas, consultas vazias, tabelas, sliders e o aviso de simulação com falha induzida apenas no teste. O modelo real também executou a consulta `ataque cardíaco` sem fallback após liberar o download no ambiente de teste. Isso verifica funcionamento, não qualidade clínica ou superioridade do ranking.

Referências das bibliotecas: [rank_bm25](https://github.com/dorianbrown/rank_bm25), [Sentence Transformers](https://www.sbert.net/docs/quickstart.html) e [cache do Streamlit](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource).
