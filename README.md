# Contagem de Veículos em Imagens Aéreas

Projeto acadêmico de Visão Computacional desenvolvido utilizando **CImg (C++)** para pré-processamento de imagens e **YOLO11n (Python)** para detecção e contagem de veículos em imagens aéreas do conjunto de dados VisDrone2019-DET.

## 1. Objetivo

O objetivo do projeto é desenvolver um sistema experimental capaz de identificar e contar veículos em imagens aéreas.

O projeto utiliza duas etapas principais:

1. **Pré-processamento:** aplicação de técnicas de processamento digital de imagens utilizando a biblioteca CImg, implementada em C++.
2. **Detecção e contagem:** utilização do modelo YOLO11n para detectar veículos, classificar os objetos e calcular a quantidade estimada em cada imagem.

O projeto considera oito categorias de veículos: bicicleta, carro, van, caminhão, triciclo, triciclo com cobertura, ônibus e motocicleta.

## 2. Tecnologias utilizadas

- C++17
- Biblioteca CImg
- Python 3
- Ultralytics YOLO11n
- PyTorch
- NumPy
- VisDrone2019-DET

O treinamento e os testes foram realizados utilizando CPU.

## 3. Estrutura do projeto

A estrutura principal está organizada da seguinte maneira:

```text
vehicle_c/
│
├── cimg/
│   ├── include/
│   │   └── CImg.h
│   ├── src/
│   │   └── main.cpp
│   ├── build/
│   │   └── cimg_preprocess
│   └── Makefile
│
├── dataset/
│   ├── VisDrone2019-DET-train/
│   │   ├── images/
│   │   └── annotations/
│   │
│   ├── dataset_yolo/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   │
│   └── processed/
│       ├── grayscale/
│       ├── gaussian/
│       ├── sobel/
│       └── equalized/
│
├── python/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── evaluate_counting.py
│
├── runs/
│   └── detect/
│       └── runs/
│           └── detect/
│               └── vehicle_counting_final/
│                   └── weights/
│                       ├── best.pt
│                       └── last.pt
│
├── results/
│   └── counting/
│       └── counting_results.csv
│
└── visdrone.yaml
```

**Observação:** a estrutura representa os principais arquivos e diretórios do projeto. Alguns diretórios são gerados durante a execução. O arquivo `best.pt` contém os melhores pesos do treinamento e é utilizado na avaliação final.

## 4. Requisitos para execução

No Linux ou WSL, instale as ferramentas de compilação:

```bash
sudo apt update
sudo apt install build-essential g++ make
```

Instale as dependências Python:

```bash
python -m pip install ultralytics numpy pyyaml
```

Verifique as instalações:

```bash
g++ --version
python --version
yolo version
```

## 5. Compilação do CImg

O módulo CImg é responsável pelas operações de pré-processamento.

A partir da raiz do projeto, execute:

```bash
make -C cimg
```

O executável esperado é:

```text
cimg/build/cimg_preprocess
```

Caso o diretório de saída ainda não exista, crie-o:

```bash
mkdir -p cimg/build
```

Se o Makefile não estiver configurado para gerar esse executável, verifique as regras de compilação nele antes de executar o programa.

## 6. Preparação do conjunto de dados

O projeto utiliza o VisDrone2019-DET.

O script `python/prepare_dataset.py` prepara as imagens e converte as anotações originais para o formato YOLO.

Execute:

```bash
python python/prepare_dataset.py
```

A preparação utilizada no experimento selecionou 500 imagens:

- 450 imagens para treinamento.
- 50 imagens para validação.

As anotações são convertidas para o formato YOLO, que utiliza a identificação da classe e as coordenadas normalizadas da caixa delimitadora.

### Classes utilizadas

| ID YOLO | Classe |
|---:|---|
| 0 | Bicycle |
| 1 | Car |
| 2 | Van |
| 3 | Truck |
| 4 | Tricycle |
| 5 | Awning-tricycle |
| 6 | Bus |
| 7 | Motor |

## 7. Pré-processamento com CImg

Depois de preparar o conjunto de dados, o executável CImg pode ser utilizado para processar as imagens de validação:

```bash
./cimg/build/cimg_preprocess \
    dataset/dataset_yolo/images/val \
    dataset/processed
```

As imagens processadas são organizadas em quatro diretórios:

| Diretório | Técnica |
|---|---|
| `grayscale/` | Conversão para escala de cinza |
| `gaussian/` | Aplicação de desfoque gaussiano |
| `sobel/` | Detecção de bordas com operador Sobel |
| `equalized/` | Equalização do histograma |

Essas técnicas permitem investigar como diferentes transformações afetam as características visuais das imagens.

**Importante:** a avaliação de contagem apresentada neste README corresponde ao modelo treinado e avaliado com o conjunto original. Não se deve afirmar que o pré-processamento melhorou o desempenho sem realizar uma comparação experimental controlada.

## 8. Modelo YOLO11n

Foi utilizado o modelo YOLO11n, uma versão compacta da arquitetura YOLO para detecção de objetos.

O treinamento final foi realizado com as seguintes configurações:

| Parâmetro | Valor |
|---|---|
| Modelo | YOLO11n |
| Épocas totais | 50 |
| Resolução | 416 × 416 |
| Dispositivo | CPU |
| Batch size | 4 |
| Otimizador | AdamW |
| Imagens de treinamento | 450 |
| Imagens de validação | 50 |

Os pesos finais estão localizados em:

```text
runs/detect/runs/detect/vehicle_counting_final/weights/
```

Arquivos:

- `best.pt`: melhores pesos encontrados durante o treinamento.
- `last.pt`: pesos do último ciclo de treinamento.

Para reproduzir as previsões, utilize `best.pt`.

## 9. Gerar imagens com as detecções

Para visualizar as detecções realizadas pelo modelo:

```bash
yolo predict \
    model="runs/detect/runs/detect/vehicle_counting_final/weights/best.pt" \
    source="dataset/dataset_yolo/images/val" \
    imgsz=416 \
    conf=0.15 \
    save=True \
    project="runs/detect" \
    name="previsao_final" \
    exist_ok=True
```

As imagens resultantes ficam em:

```text
runs/detect/previsao_final/
```

Elas apresentam as caixas delimitadoras, as classes identificadas e os níveis de confiança das detecções.

## 10. Avaliação da detecção

A avaliação de detecção utiliza métricas tradicionais de modelos de detecção de objetos.

Resultados obtidos na validação final:

| Métrica | Resultado |
|---|---:|
| Precisão (Precision) | 23,2% |
| Revocação (Recall) | 21,6% |
| mAP@0.5 | 16,8% |
| mAP@0.5:0.95 | 8,65% |

### Significado das métricas

**Precisão (Precision)**

Indica a proporção de detecções realizadas pelo modelo que realmente correspondem a objetos anotados.

Uma precisão maior significa menos falsos positivos.

**Revocação (Recall)**

Indica a proporção dos objetos reais que foram identificados pelo modelo.

Uma revocação maior significa que o modelo deixou de detectar menos objetos.

**mAP@0.5**

Mean Average Precision considerando uma interseção sobre união (IoU) mínima de 0,5 entre a caixa prevista e a caixa real.

**mAP@0.5:0.95**

Calcula a média do mAP em diferentes limiares de IoU, de 0,5 até 0,95, com incrementos de 0,05. É uma avaliação mais rigorosa da qualidade das caixas delimitadoras.

**Atenção:** mAP não representa a porcentagem de acerto da contagem de veículos. Para avaliar a contagem, são utilizadas métricas específicas, descritas na próxima seção.

## 11. Avaliação da contagem de veículos

O script `python/evaluate_counting.py` compara a quantidade de veículos prevista pelo YOLO com a quantidade real registrada nas anotações do conjunto de validação.

Execute:

```bash
python python/evaluate_counting.py
```

O script realiza as seguintes operações:

1. Carrega os pesos `best.pt`.
2. Percorre as 50 imagens de validação.
3. Consulta os arquivos de anotação correspondentes.
4. Conta os objetos reais de cada imagem.
5. Executa a detecção pelo YOLO.
6. Conta os objetos previstos.
7. Calcula as métricas de contagem.
8. Salva a comparação em um arquivo CSV.

A avaliação utiliza confiança mínima (`conf`) de 0,15.
### Métricas de contagem

Para avaliar a capacidade do modelo de estimar corretamente a quantidade de veículos em cada imagem, foram utilizadas as métricas MAE, RMSE, Bias, WAPE e Contagem Exata.

**1. MAE — Mean Absolute Error (Erro Absoluto Médio)**

Calcula a média das diferenças absolutas entre a quantidade prevista e a quantidade real de veículos.

$$
MAE = \frac{1}{N}\sum_{i=1}^{N}|P_i-R_i|
$$

**2. RMSE — Root Mean Squared Error (Raiz do Erro Quadrático Médio)**

Calcula a raiz da média dos erros quadráticos. Essa métrica penaliza mais fortemente erros de contagem elevados.

$$
RMSE = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(P_i-R_i)^2}
$$

**3. Bias — Viés Médio**

Indica se o modelo apresenta tendência a superestimar ou subestimar a quantidade de veículos.

$$
Bias = \frac{1}{N}\sum_{i=1}^{N}(P_i-R_i)
$$

- **Bias positivo:** o modelo tende a prever mais veículos do que existem.
- **Bias negativo:** o modelo tende a prever menos veículos do que existem.
- **Bias próximo de zero:** indica que os erros positivos e negativos se compensam na média.

**4. WAPE — Weighted Absolute Percentage Error (Erro Percentual Absoluto Ponderado)**

Representa a soma dos erros absolutos em relação à quantidade total de veículos reais, expressa em porcentagem.

$$
WAPE = \frac{\sum_{i=1}^{N}|P_i-R_i|}{\sum_{i=1}^{N}R_i}\times 100
$$

**5. Contagem Exata**

Representa o percentual de imagens em que a quantidade prevista pelo modelo foi exatamente igual à quantidade real.

$$
Contagem\ Exata =
\frac{\text{Número de imagens com contagem correta}}{N}
\times 100
$$

**Variáveis utilizadas:**

- \(P_i\): quantidade de veículos prevista pelo modelo na imagem \(i\).
- \(R_i\): quantidade real de veículos registrada na anotação da imagem \(i\).
- \(N\): número total de imagens avaliadas.

**Interpretação:** para MAE, RMSE e WAPE, valores menores indicam melhor desempenho. Para a contagem exata, valores maiores indicam maior proporção de imagens com estimativas corretas. O Bias deve ser analisado considerando seu sinal e sua magnitude.

### Resultados obtidos

Avaliação realizada em 50 imagens de validação:

| Métrica | Resultado |
|---|---:|
| Veículos reais | 2.139 |
| Veículos previstos | 2.171 |
| MAE | 12,28 |
| RMSE | 16,45 |
| Viés médio | +0,64 |
| WAPE | 28,71% |
| Contagem exata | 4,00% |

Os resultados mostram que o total previsto ficou próximo do total real. Entretanto, o erro médio absoluto demonstra que ainda existem diferenças consideráveis entre as contagens individuais das imagens.

A proximidade dos totais não significa que o modelo tenha contado corretamente cada imagem.

### Resultados por classe

| Classe | Real | Previsto | MAE | WAPE |
|---|---:|---:|---:|---:|
| Bicycle | 121 | 0 | 2,42 | 100,00% |
| Car | 1.138 | 1.798 | 14,16 | 62,21% |
| Van | 187 | 28 | 3,34 | 89,30% |
| Truck | 59 | 50 | 1,26 | 106,78% |
| Tricycle | 96 | 16 | 1,72 | 89,58% |
| Awning-tricycle | 37 | 0 | 0,74 | 100,00% |
| Bus | 18 | 0 | 0,36 | 100,00% |
| Motor | 483 | 279 | 5,20 | 53,83% |

O modelo apresentou melhor capacidade de identificação da classe Car, enquanto outras categorias tiveram dificuldades significativas de detecção.

O WAPE de uma classe pode ultrapassar 100% quando a quantidade de erros absolutos supera a quantidade real daquela classe.

## 12. Arquivos de resultados

Os principais arquivos para consulta e elaboração do relatório são:

| Arquivo ou diretório | Conteúdo |
|---|---|
| `cimg/src/main.cpp` | Implementação do pré-processamento em C++ |
| `cimg/Makefile` | Configuração de compilação |
| `python/prepare_dataset.py` | Preparação e conversão do conjunto de dados |
| `python/train.py` | Configuração do treinamento |
| `python/evaluate_counting.py` | Avaliação da contagem |
| `visdrone.yaml` | Configuração do conjunto de dados para YOLO |
| `runs/detect/.../weights/best.pt` | Pesos finais do modelo |
| `runs/detect/previsao_final/` | Imagens com as detecções |
| `results/counting/counting_results.csv` | Comparação da contagem por imagem |

## 13. Limitações e trabalhos futuros

O experimento apresentou limitações relacionadas à quantidade de imagens utilizadas, ao desbalanceamento entre as classes e à dificuldade de detectar objetos pequenos ou parcialmente ocultos.

Como possibilidades de melhoria, destacam-se:

- Aumentar a quantidade de imagens de treinamento.
- Melhorar a distribuição de amostras entre as classes.
- Investigar diferentes resoluções de entrada.
- Comparar o desempenho com e sem pré-processamento.
- Avaliar outras configurações de confiança.
- Investigar arquiteturas específicas para contagem de objetos em imagens aéreas.

## 14. Considerações finais

O projeto implementou um fluxo de processamento de imagens aéreas utilizando CImg e YOLO11n, permitindo realizar a detecção e a estimativa da quantidade de veículos.

A avaliação foi dividida em duas partes: qualidade das detecções, utilizando Precision, Recall e mAP; e qualidade da contagem, utilizando MAE, RMSE, Bias, WAPE e percentual de contagem exata.

Os resultados demonstram a execução do fluxo proposto e também evidenciam limitações que podem ser exploradas em trabalhos futuros.
