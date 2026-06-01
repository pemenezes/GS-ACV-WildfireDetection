# GS-ACV-WildfireDetection
Repo destinated to the GS work for FIAP


# 🔥 Wildfire Detection — Global Solution FIAP 2026

**Curso:** Engenharia de Software — 4º Ano  
**Disciplina:** Applied Computer Vision

**Integrantes:** Anny Dias — RM98295

**Integrantes:** Henrique Lima — RM551528

**Integrantes:** Pedro Gava — RM551043

**Integrantes:** Pedro Menezes — RM97432

---

## Problema
Detecção automática de queimadas em imagens de satélite usando
redes neurais convolucionais treinadas do zero, aplicada ao
monitoramento ambiental no contexto da Indústria Espacial.

**ODS:** 13 — Ação Climática | 15 — Vida Terrestre

---

## Dataset
- Fonte: [Wildfire Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
- Classes: `wildfire` / `nowildfire`
- Total: 42.850 imagens (train: 30.250 | val: 6.300 | test: 6.300)
- Resolução: 224×224px

---

## Modelos treinados

| Modelo | Épocas | Acc Validação | Acc Teste |
|--------|--------|--------------|-----------|
| SimpleFireNet | 25 | 96,98% | **97,84%** |
| DeepFireNet | 11 | 95,92% | 96,76% |

**Melhor modelo:** SimpleFireNet  
A SimpleFireNet apresentou convergência mais estável e superou
a DeepFireNet em todas as métricas. A DeepFireNet sofreu
oscilações durante o treino (val acc caiu para 78% na época 2),
indicando sensibilidade ao learning rate com 3 blocos convolucionais.

---

## Como executar

1. Abra o notebook no Google Colab
2. Baixe as seguintes bibliotecas: torch, torchvision, Pillow, matplotlib, scikit-learn, gradio
3. Ative a GPU: `Ambiente de execução > T4 GPU`
4. Execute todas as células em ordem
5. O dataset é baixado automaticamente via Kaggle API

---

## Demonstração
🔗 [Aplicação Gradio](https://huggingface.co/spaces/pemenezzz/wildfire-detection)  
🎥 [Vídeo demonstrativo](https://youtu.be/q9VT3i_8AEY)

---

## Limitações
O modelo apresentou dificuldade com imagens de altíssima altitude
orbital (ex: imagens Maxar em escala de cidade inteira), pois o
dataset de treino é composto majoritariamente por imagens aéreas
em altitude média. Como trabalho futuro, incluir imagens Sentinel-2
do INPE no dataset aumentaria a robustez para esse cenário.

---
Dataset usado para testes:
[img_para_teste.zip](https://github.com/user-attachments/files/28483141/img_para_teste.zip)
