# Predição de Diabetes com Modelos Supervisionados

Este projeto implementa e compara diferentes abordagens supervisionadas para predição do risco de diabetes mellitus, com base em dados clínicos simples. São utilizadas abordagens estatísticas (regressão logística) e computacionais (Perceptron, MLP), com análise completa de desempenho, calibração e viabilidade interpretativa.

## 🔍 Objetivo

Avaliar o desempenho e a interpretabilidade de diferentes modelos supervisionados aplicados ao conjunto de dados de diabetes do repositório UCI. As abordagens incluem:

- **Regressão Logística (Estatística Clássica)**  
- **Regressão Logística (ML)**  
- **Perceptron**  
- **Multilayer Perceptron (MLP)**  

## 🗂️ Organização dos Arquivos

| Arquivo                         | Descrição                                                                 |
|--------------------------------|---------------------------------------------------------------------------|
| `regressao_logistica.ipynb`    | Abordagem estatística clássica com ajuste linear, splines e calibração.  |
| `perceptron_mlp_cnn.ipynb`     | Abordagem computacional com Perceptron e MLP implementados em Keras.     |
| `multilayer_perceptron.ipynb`  | Treinamento e avaliação detalhada de uma MLP com duas camadas ocultas.   |
| `mlp.ipynb`                     | Script alternativo com foco na arquitetura e visualização de métricas.   |
| `aed.ipynb`                     | Análise exploratória dos dados (limpeza, transformação e visualização).  |
| `Supervised_Learning_All_Days.pdf` | Material teórico de apoio (ML, VC-dim, otimização, teoria dos modelos). |

## 📊 Base de Dados

Conjunto `Pima Indians Diabetes` da UCI, com 768 observações e 9 variáveis clínicas.  
Após limpeza e exclusão de valores impossíveis (ex: IMC = 0), foram mantidas 392 observações.

### Preditores utilizados:

- **Glucose** – Glicemia de jejum  
- **BMI** – Índice de Massa Corporal  
- **Age** – Idade  
- **Pregnancies** – Número de gestações  

Estes foram escolhidos com base em significância estatística, disponibilidade prática e coerência clínica.

## ⚙️ Modelos Implementados

### 🔹 Regressão Logística (Estatística)
- Implementada em R com `glm()` e `rms::lrm()`
- Calibração via Hosmer–Lemeshow e `bootstrap`
- Curva de calibração gerada para análise de ajuste
- Interpretabilidade direta por coeficientes

### 🔹 Multilayer Perceptron (MLP)
- Implementado com Keras/TensorFlow
- Arquitetura: [8 entradas] → [6] → [3] → [2 saídas, Softmax]
- Otimização com `gradient descent` (batch, 1.500 épocas)
- Ativação ReLU nas camadas ocultas

### 🔹 Perceptron
- Classificador linear simples
- Acompanhado de estudo teórico e histórico

## 📈 Avaliação de Desempenho

| Métrica           | Linear (Validação) | MLP (Validação) |
|------------------|--------------------|------------------|
| Acurácia          | 0,831              | 0,831            |
| AUC               | 0,863              | 0,869            |
| Sensibilidade     | 0,872              | 0,897            |
| Especificidade    | 0,750              | 0,700            |
| Balanced Accuracy | 0,811              | 0,799            |

- O modelo linear apresentou desempenho comparável ao MLP, com vantagem interpretativa e menor risco de sobreajuste.

## 📌 Conclusões

- A **Regressão Logística Linear** com recalibração demonstrou excelente desempenho preditivo e interpretabilidade clínica.
- O modelo **MLP** apresentou leve vantagem em AUC, mas com maior custo computacional e menor especificidade.
- O modelo linear é preferível para triagem populacional, especialmente em cenários com recursos computacionais limitados.

## 📚 Referências

O material teórico e metodológico está documentado no PDF [`Supervised_Learning_All_Days.pdf`](Supervised_Learning_All_Days.pdf), incluindo:

- História e limites do Perceptron
- Teoria da VC-dimensão e generalização
- Otimização com descida de gradiente
- Comparação entre modelos estatísticos e computacionais

## 👨‍🔬 Autoria

Projeto desenvolvido por Pedro Brom, George Fabrício, Charles Santos e Alexandro de Paula  
Programa de Pós-Graduação em Informática (PPGI) – Universidade de Brasília

---

