# HealthSearch

Projeto acadêmico em Python e Streamlit que demonstra busca híbrida em uma pequena base de documentos sobre saúde.

O aplicativo compara três formas de busca:

- **BM25:** busca por palavras-chave.
- **Semântica simulada:** usa conceitos e sinônimos definidos manualmente, com similaridade de cosseno. Não utiliza um modelo treinado.
- **Híbrida (RRF):** combina as posições dos resultados das duas buscas.

## Como executar

Com o Python e o pip instalados, execute na pasta do projeto:

```bash
python -m pip install -r requirements.txt
python -m streamlit run healthsearch_app.py
```

Abra o endereço exibido no terminal, normalmente `http://localhost:8501`.

## Como usar

Digite uma consulta, como `COD-ECG-12D`, `derrame` ou `dor no peito`, e confira os resultados nas abas de busca e comparação.

Na barra lateral, ajuste `k1` e `b` para configurar o BM25 e `alpha` para alterar o peso de cada busca na combinação: `1` prioriza o BM25 e `0` prioriza a busca semântica.

Os seis documentos de exemplo estão definidos no próprio arquivo `healthsearch_app.py`.
