# 🔥 Wildfire Detection — Global Solution FIAP 2026

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![Accuracy](https://img.shields.io/badge/Acurácia-97.84%25-brightgreen)
![ODS](https://img.shields.io/badge/ODS-13%20|%2015-green)

**Curso:** Engenharia de Software — 4º Ano  
**Disciplina:** Applied Computer Vision  
**Instituição:** FIAP — Global Solution 2026  

| Integrante | RM |
|---|---|
| Anny Dias | RM98295 |
| Henrique Lima | RM551528 |
| Pedro Gava | RM551043 |
| Pedro Menezes | RM97432 |

---

## Problema

Detecção automática de queimadas em imagens de satélite usando redes neurais
convolucionais treinadas do zero, aplicada ao monitoramento ambiental no
contexto da Indústria Espacial.

> Dados orbitais e imagens aéreas são hoje uma das principais ferramentas de
> monitoramento ambiental em escala global. Este projeto aplica visão
> computacional para classificar automaticamente áreas com queimadas ativas,
> contribuindo para sistemas de alerta precoce.

**ODS conectados:** 13 — Ação Climática | 15 — Vida Terrestre

---

## Dataset

- **Fonte:** [Wildfire Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset)
- **Classes:** `wildfire` / `nowildfire`
- **Distribuição:**

| Split | wildfire | nowildfire | Total |
|-------|----------|------------|-------|
| Train | 15.750 | 14.500 | 30.250 |
| Val | 3.480 | 2.820 | 6.300 |
| Test | 3.480 | 2.820 | 6.300 |
| **Total** | | | **42.850** |

- **Resolução:** 224×224px  
- **Pré-processamento:** resize, normalização ImageNet, data augmentation no treino
(flip horizontal/vertical, rotação ±15°, color jitter)

---

## Arquitetura dos Modelos

Dois modelos CNN treinados do zero — sem uso de modelos pré-treinados.

### SimpleFireNet

Input (3×224×224)
→ Conv2d(3→32, 3×3) → ReLU → MaxPool2d(2×2)
→ Conv2d(32→64, 3×3) → ReLU → MaxPool2d(2×2)
→ Flatten → Linear(200704→256) → ReLU → Dropout(0.5)
→ Linear(256→2)

### DeepFireNet

Input (3×224×224)
→ Conv2d(3→32) → BatchNorm2d → ReLU → MaxPool2d
→ Conv2d(32→64) → BatchNorm2d → ReLU → MaxPool2d
→ Conv2d(64→128) → BatchNorm2d → ReLU → MaxPool2d
→ Flatten → Linear(→512) → ReLU → Dropout(0.4)
→ Linear(512→256) → ReLU → Dropout(0.4)
→ Linear(256→2)

---

## Resultados

| Modelo | Épocas | Acc Validação | Acc Teste | F1 médio |
|--------|--------|--------------|-----------|----------|
| SimpleFireNet | 25 | 96,98% | **97,84%** | **0,98** |
| DeepFireNet | 11 | 95,92% | 96,76% | 0,97 |

**Melhor modelo: SimpleFireNet**

A SimpleFireNet apresentou convergência estável ao longo das 25 épocas.
A DeepFireNet, apesar de mais profunda e com Batch Normalization, sofreu
oscilações durante o treino — a acurácia de validação caiu para 78% na
época 2 e o early stopping foi acionado na época 11, indicando
sensibilidade ao learning rate com 3 blocos convolucionais.

### Curvas de treino

**SimpleFireNet**  
<img width="900" alt="historico_SimpleFireNet"
src="https://github.com/user-attachments/assets/d55a5400-6f87-4b9f-9004-a05644d175fb" />

**DeepFireNet**  
<img width="900" alt="historico_DeepFireNet"
src="https://github.com/user-attachments/assets/cac5724d-94b2-41c2-ad55-6abc46c97440" />

### Matrizes de confusão

<p align="left">
<img width="380" alt="confusion_SimpleFireNet"
src="https://github.com/user-attachments/assets/88e785b9-42b1-49ce-85d7-3fc0b645a7c0" />
<img width="380" alt="confusion_DeepFireNet"
src="https://github.com/user-attachments/assets/c08622eb-f318-4c69-97f0-14e505007a88" />
</p>

---

## Demonstração

🔗 [Aplicação no Hugging Face Spaces](https://huggingface.co/spaces/pemenezzz/wildfire-detection)  
🎥 [Vídeo demonstrativo — YouTube](https://youtu.be/q9VT3i_8AEY)

<img width="900" alt="demo_gradio"
src="https://github.com/user-attachments/assets/e7e1b9cf-2c77-4ab5-a097-b4ea9a501ada" />

---

## Como executar

### Pré-requisitos

- Conta no **Google** (para usar o Google Colab + Google Drive)
- Conta no **Kaggle** com token de API gerado

### Passo a passo

**1. Obter o token da Kaggle API**

Acesse `kaggle.com` → Settings → API Tokens → **Create New API Token**.  
Isso gera o seu `username` e `key` pessoais.

**2. Abrir o notebook no Google Colab**

Abra o arquivo `GS_ACV.ipynb` no Google Colab.

**3. Ativar a GPU**

`Ambiente de execução → Alterar tipo de ambiente de execução → T4 GPU`

**4. Executar as células em sequência**

> ⚠️ **Atenção:** as células abaixo precisam ser configuradas com seus próprios dados antes de executar.

**Célula de configuração do Kaggle** — substitua com seu username e key:
```python
kaggle_config = {
    "username": "SEU_USERNAME_KAGGLE",  # ← substitua aqui
    "key": "SUA_KEY_KAGGLE"             # ← substitua aqui
}
```

**Célula de montagem do Drive** — ao executar, autorize o acesso à **sua** conta Google.  
Os modelos treinados serão salvos na sua Drive em `/GS_ACV/`.

**5. Sobre o treinamento**

As células de treino estão **comentadas** no notebook porque os modelos já foram treinados.  
Para re-treinar do zero, descomente as linhas finais da célula de treino:

```python
historico_simple = treinar_modelo(SimpleFireNet(NUM_CLASSES), "SimpleFireNet")
historico_deep   = treinar_modelo(DeepFireNet(NUM_CLASSES),   "DeepFireNet")
```

> ⚠️ O treino completo leva aproximadamente **3 a 4 horas** com GPU T4.  
> Sem re-treinar, as células de avaliação e o Gradio não funcionarão pois os  
> arquivos `.pth` estão no Drive do autor original.

**6. Dependências**

Instaladas automaticamente pelo Colab: `torch` `torchvision` `Pillow` `matplotlib` `scikit-learn` `gradio`

---

## Testar o modelo no gradio

Para testar o modelo localmente com imagens próprias, use o arquivo:  
📦 [img_para_teste.zip](https://github.com/user-attachments/files/28483141/img_para_teste.zip)

---

## Acessar arquivo com pesos do modelo

para acessar o arquivo que contem os pesos do melhor modelo :
https://drive.google.com/file/d/1dzgASUZcRobdMdYHts153cFQl6ha-GMr/view?usp=sharing

---

## Limitações

O modelo apresentou dificuldade com imagens de altíssima altitude orbital
(ex: imagens Maxar em escala de cidade inteira), pois o dataset de treino
é composto majoritariamente por imagens aéreas em altitude média.

Como trabalho futuro, incluir imagens Sentinel-2 do INPE no dataset
aumentaria a robustez para esse cenário e daria maior representatividade
ao contexto brasileiro de monitoramento de queimadas no Cerrado e Amazônia.
