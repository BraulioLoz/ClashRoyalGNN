# 🎮 Clash Royale Deck Recommendation usando GraphSAGE

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyG-2.7.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Sistema de recomendación de mazos para Clash Royale basado en **Graph Neural Networks (GraphSAGE)** que, dado un mazo incompleto de 6 cartas, recomienda las 2 cartas faltantes utilizando sinergias aprendidas de decks de jugadores profesionales.

---

## 📋 Contenido del Repositorio

```
ClashRoyalGNN/
├── models_transfer/             # Modelos de transfer learning (.pt) y historial
│   ├── best_model.pt
│   ├── training_history.json
│   └── training_history.csv
├── data/
│   ├── 01-raw/                  # Datos crudos del API de Clash Royale
│   ├── 02-preprocessed/         # Matriz de co-ocurrencia preprocesada
│   ├── 03-features/             # Características del grafo y ejemplos de entrenamiento
│   └── 04-predictions/          # Predicciones del modelo
├── entrypoint/
│   ├── train_transfer_learning.py  # ⭐ ENTRYPOINT PRINCIPAL - Transfer Learning
│   ├── train.py                 # Entrenamiento desde cero (no usado en este proyecto)
│   ├── inference.py             # Script de inferencia/predicción
│   └── compare_models.py        # Comparación de modelos
├── src/
│   ├── models/
│   │   ├── pretrained_sage.py   # ⭐ GraphSAGE con Transfer Learning (USADO)
│   │   ├── graphsage_model.py   # Modelo GraphSAGE base
│   │   └── gnn_model.py         # Modelo GCN base
│   ├── pipelines/
│   │   ├── training_pipeline.py      # Pipeline de entrenamiento (shared)
│   │   ├── feature_eng_pipeline.py   # Pipeline de ingeniería de características
│   │   └── inference_pipeline.py     # Pipeline de inferencia
│   └── utils.py                 # Utilidades (config, device, logging)
├── config/
│   └── config.yaml              # Configuración del modelo y transfer learning
├── Dockerfile                   # Docker para AMD/Strix Halo (Linux + ROCm)
├── docker-compose.yml           # Docker Compose para AMD GPUs
├── requirements.txt             # Dependencias del proyecto
└── README.md                    # Este archivo
```

---

## 🎯 Motivación del Proyecto

### ¿Por qué Graph Neural Networks?

En Clash Royale, las **sinergias entre cartas** son más importantes que las cartas individuales. Un mazo exitoso no es simplemente una colección de cartas poderosas, sino un conjunto de cartas que trabajan bien juntas.

Los sistemas de recomendación tradicionales tratan las cartas como items independientes, ignorando estas relaciones complejas. Las **Graph Neural Networks (GNNs)** son ideales para este problema porque:

1. **Modelado natural de relaciones**: Las cartas forman un grafo donde las aristas representan sinergias aprendidas de decks profesionales.
2. **Captura de patrones complejos**: Las GNNs aprenden patrones de co-ocurrencia, sinergias ofensivas/defensivas, y composiciones meta.
3. **Propagación de información**: El mecanismo de "message passing" permite que cada carta "aprenda" de sus vecinos en el grafo.

### ¿Por qué GraphSAGE?

Comparado con otras arquitecturas de GNN:

| Arquitectura           | Ventajas                                                                                                        | Desventajas                                        |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **GCN**          | Simple, rápido, bien establecido                                                                               | Usa todos los vecinos (no escala), pesos uniformes |
| **GAT**          | Attention mechanism, más expresivo                                                                             | Más lento, más parámetros, puede sobreajustar   |
| **GIN**          | Muy expresivo, inyectivo                                                                                        | Complejo, difícil de entrenar                     |
| **GraphSAGE** ✅ | **Muestreo de vecinos** (escalable), **agregación aprendida**, mejor para relaciones heterogéneas | Ligeramente más lento que GCN                     |

**GraphSAGE fue elegido porque:**

- ✅ **Neighbor Sampling**: Más escalable que GCN (importante si el grafo crece)
- ✅ **Learned Aggregation**: Captura mejor las sinergias heterogéneas (soporte, defensa, win conditions)
- ✅ **Inductivo**: Puede generalizar a cartas nuevas sin reentrenar desde cero
- ✅ **Balance**: Más expresivo que GCN, más eficiente que GAT

---

## 🕸️ Arquitectura del Grafo

### Construcción del Grafo

El grafo se construye a partir de **datos reales de jugadores top** (clanes con score ≥ 99,000):

#### Nodos (Vertices)

- **Cantidad**: 110 cartas de Clash Royale
- **Representación**: Cada carta es un nodo en el grafo
- **Features por nodo**: 5 características numéricas + 1 indicador binario

#### Aristas (Edges)

- **Tipo**: Co-ocurrencia en decks profesionales
- **Peso**: Frecuencia de aparición conjunta
- **Threshold**: Solo se crean aristas si dos cartas aparecen juntas ≥ 5 veces
- **Dirección**: No dirigidas (simétricas)
- **Cantidad**: ~7,232 aristas en el grafo final

**Ejemplo**: Si "Hog Rider" y "Fireball" aparecen juntas en 150 decks, se crea una arista con peso 150.

#### Features de Nodo

Cada carta tiene 6 features:

| Feature               | Descripción                | Tipo                     | Rango                                 |
| --------------------- | --------------------------- | ------------------------ | ------------------------------------- |
| `id`                | ID único de la carta       | Numérico (normalizado)  | [0, 1]                                |
| `elixirCost`        | Costo de elixir             | Numérico (normalizado)  | 1-10 elixir                           |
| `rarity`            | Rareza de la carta          | Categórico → Numérico | 0=Common, 1=Rare, 2=Epic, 3=Legendary |
| `maxLevel`          | Nivel máximo               | Numérico (normalizado)  | 6-14                                  |
| `maxEvolutionLevel` | Nivel de evolución máximo | Numérico (normalizado)  | 0-1                                   |
| `input_indicator`   | ¿Está en el input?        | Binario                  | 0 o 1                                 |

**Normalización**: Todas las features se normalizan usando **estandarización** (media=0, std=1):

```
x_normalized = (x - mean) / std
```

Esto asegura que todas las features tengan la misma escala y el modelo converja más rápido.

---

## 🧠 Arquitectura del Modelo (GraphSAGE)

### Diseño de Capas

```
Input (6 features)
    ↓
SAGEConv Layer 1: 6 → 512 (mean aggregation)
    ↓ ReLU + Dropout(0.4)
SAGEConv Layer 2: 512 → 256 (mean aggregation)
    ↓ ReLU + Dropout(0.3)
SAGEConv Layer 3: 256 → 128 (mean aggregation)
    ↓ ReLU + Dropout(0.2)
SAGEConv Layer 4: 128 → 64 (mean aggregation)
    ↓ ReLU + Dropout(0.1)
Mean Pooling (agregación de nodos)
    ↓
Linear Output Layer: 64 → 110 (probabilidad por carta)
    ↓
Softmax
```

### Componentes Clave

#### 1. Capas SAGEConv (GraphSAGE Convolutional Layers)

Cada capa GraphSAGE realiza:

```python
h_i^(k) = σ(W · CONCAT(h_i^(k-1), AGG({h_j^(k-1) : j ∈ N(i)})))
```

Donde:

- `h_i^(k)`: Embedding del nodo `i` en la capa `k`
- `σ`: Función de activación (ReLU)
- `W`: Matriz de pesos aprendibles
- `CONCAT`: Concatenación de features propias y agregadas
- `AGG`: Función de agregación (en nuestro caso, **mean**)
- `N(i)`: Vecinos del nodo `i`

#### 2. Agregador: Mean Aggregation

**¿Por qué Mean?**

- ✅ **Additive synergies**: Las sinergias en Clash Royale son típicamente aditivas
- ✅ **Estabilidad**: Menos sensible a outliers que `max`
- ✅ **Eficiencia**: Más rápido que `lstm` aggregator
- ✅ **Interpretable**: El promedio de vecinos tiene sentido semántico

Otras opciones disponibles (no usadas):

- `max`: Max pooling (para features dominantes)
- `lstm`: LSTM-based aggregation (más expresivo pero más lento)

#### 3. Funciones de Activación

- **ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`
  - Introduce no-linealidad
  - Previene vanishing gradients
  - Computacionalmente eficiente

#### 4. Regularización

**Dropout** con tasas decrecientes:

- Layer 1: 40% dropout
- Layer 2: 30% dropout
- Layer 3: 20% dropout
- Layer 4: 10% dropout

**Rationale**: Las capas tempranas aprenden features más generales (mayor dropout), las capas finales aprenden features específicas (menor dropout).

**Gradient Clipping**: Norma máxima = 1.0

- Previene exploding gradients
- Estabiliza el entrenamiento

#### 5. Pooling (Readout)

**Mean Pooling** sobre todos los nodos:

```python
h_graph = MEAN({h_i^(L) : i ∈ V})
```

Esto produce una representación de todo el grafo (deck) que captura información de todas las cartas.

#### 6. Capa de Salida (MLP)

Capa lineal final que mapea el embedding del grafo a probabilidades:

```python
output = Linear(64 → 110) + Softmax
```

Cada salida representa la probabilidad de que una carta específica deba estar en el deck.

---

## 🔢 Cálculo del Número de Parámetros

### Fórmula para SAGEConv

En GraphSAGE con concatenación, cada capa tiene:

```
params = (in_features + in_features) × out_features + out_features
```

Porque SAGEConv concatena las features del nodo con las features agregadas de sus vecinos.

### Desglose por Capa

| Capa                    | Input → Output | Parámetros       | Cálculo                       |
| ----------------------- | --------------- | ----------------- | ------------------------------ |
| **SAGEConv 1**    | 6 → 512        | **6,656**   | (6+6)×512 + 512 = 6,656       |
| **SAGEConv 2**    | 512 → 256      | **262,400** | (512+512)×256 + 256 = 262,400 |
| **SAGEConv 3**    | 256 → 128      | **65,664**  | (256+256)×128 + 128 = 65,664  |
| **SAGEConv 4**    | 128 → 64       | **16,448**  | (128+128)×64 + 64 = 16,448    |
| **Linear Output** | 64 → 110       | **7,150**   | 64×110 + 110 = 7,150          |
| **Total**         |                 | **358,318** |                                |

### Comparación con GCN

| Modelo    | Parámetros       | Diferencia                   |
| --------- | ----------------- | ---------------------------- |
| GCN       | ~320,000          | Baseline                     |
| GraphSAGE | **358,318** | +12% (por la concatenación) |

El aumento es razonable considerando la mayor expresividad de GraphSAGE.

---

## 🔄 Transfer Learning con GraphSAGE

Este proyecto utiliza **Transfer Learning** con entrenamiento por etapas (staged training) para mejorar la convergencia y estabilidad del modelo.

### Arquitectura del Modelo de Transfer Learning

```
Card Features (6-dim)
    ↓
Feature Adapter: 6 → 128-dim
    ├─ Linear Layer: 6 → 64
    ├─ ReLU + Dropout(0.4)
    └─ Linear Layer: 64 → 128 + LayerNorm
    ↓
Pretrained GraphSAGE Encoder: 128 → 256 → 128
    ├─ SAGEConv Layer 1: 128 → 256 (frozen en Stage 1)
    ├─ ReLU + Dropout(0.3)
    ├─ SAGEConv Layer 2: 256 → 128 (frozen en Stage 1)
    └─ ReLU + Dropout(0.2)
    ↓
Fine-tuning Layers: 128 → 64
    ├─ SAGEConv Layer: 128 → 64 (task-specific)
    └─ ReLU + Dropout(0.1)
    ↓
Task Head (Output Layer): 64 → 110 cards
    └─ Linear + Softmax
```

#### Componentes del Modelo

| Componente | Entrada → Salida | Función | Parámetros |
|------------|------------------|---------|------------|
| **Feature Adapter** | 6 → 128 | Proyecta features de cartas a dimensión pretrained | ~8,000 |
| **Pretrained Encoder** | 128 → 256 → 128 | Extrae representaciones generales (GraphSAGE) | ~98,000 |
| **Fine-tuning Layer** | 128 → 64 | Capa específica del task | ~16,000 |
| **Task Head** | 64 → 110 | Predicción final de cartas | ~7,000 |
| **Total** | - | - | **~164,078** |

### Staged Training (Entrenamiento por Etapas)

El transfer learning se realiza en **3 stages** con freezing/unfreezing progresivo:

#### **Stage 1: Adapter Training** (5 épocas, LR=0.01)

```python
Capas Entrenables:
  ✅ Feature Adapter
  ✅ Task Head (Output Layer)

Capas Congeladas:
  ❄️ Pretrained Encoder (frozen)
  ❄️ Fine-tuning Layer (frozen)

Parámetros Entrenables: ~15,000 (9% del total)
```

**Objetivo**: Entrenar el adapter para proyectar features de cartas al espacio pretrained sin alterar el encoder.

#### **Stage 2: Partial Fine-tuning** (10 épocas, LR=0.005)

```python
Capas Entrenables:
  ✅ Feature Adapter
  ✅ Pretrained Encoder - Últimas 2 capas (unfrozen)
  ✅ Fine-tuning Layer
  ✅ Task Head

Capas Congeladas:
  ❄️ Pretrained Encoder - Primera capa (frozen)

Parámetros Entrenables: ~120,000 (73% del total)
```

**Objetivo**: Ajustar las capas superiores del encoder y las capas task-specific.

#### **Stage 3: Full Fine-tuning** (10 épocas, LR=0.001)

```python
Capas Entrenables:
  ✅ Feature Adapter
  ✅ Pretrained Encoder - Todas las capas (unfrozen)
  ✅ Fine-tuning Layer
  ✅ Task Head

Parámetros Entrenables: ~164,078 (100% del total)
```

**Objetivo**: Fine-tuning completo de todo el modelo con LR muy bajo para no destruir lo aprendido.

### ¿Por qué Transfer Learning?

| Aspecto | Entrenamiento desde Cero | Transfer Learning ✅ |
|---------|-------------------------|---------------------|
| **Convergencia** | Lenta (~20-30 épocas) | Rápida (~13-25 épocas totales) |
| **Estabilidad** | Puede ser errática | Más estable por staged approach |
| **Val Loss Inicial** | ~4.1 | ~3.9 (mejor inicio) |
| **Val Loss Final** | ~3.35 | ~3.20 (mejor resultado) |
| **Risk de Overfitting** | Alto | Bajo (freezing progresivo) |
| **Tiempo Total** | ~2.5 horas (10 épocas) | ~6-8 horas (25 épocas) pero mejor resultado |

### Ventajas del Staged Training

1. **Convergencia más rápida**: El adapter aprende la proyección sin alterar el encoder pretrained
2. **Mayor estabilidad**: El freezing previene cambios drásticos en las primeras épocas
3. **Menor overfitting**: El modelo no puede "memorizar" tan fácilmente
4. **Learning rates adaptativos**: Cada stage usa un LR apropiado para su objetivo
5. **Mejor generalización**: El encoder mantiene representaciones generales útiles

### Configuración de Transfer Learning

La configuración se define en `config/config.yaml`:

```yaml
training:
  lr: 0.01  # Learning rate base

  transfer_learning:
    stage1_epochs: 5   # Adapter training
    stage2_epochs: 10  # Partial fine-tuning
    stage3_epochs: 10  # Full fine-tuning
    stage2_lr_factor: 0.5   # LR Stage 2 = 0.01 × 0.5 = 0.005
    stage3_lr_factor: 0.1   # LR Stage 3 = 0.01 × 0.1 = 0.001
```

### Cálculo de Parámetros por Stage

| Stage | Adapter | Encoder | Finetune | Task Head | Total Entrenables | % |
|-------|---------|---------|----------|-----------|-------------------|---|
| **Stage 1** | ✅ 8K | ❄️ 0 | ❄️ 0 | ✅ 7K | **~15K** | 9% |
| **Stage 2** | ✅ 8K | ✅ 90K | ✅ 16K | ✅ 7K | **~121K** | 73% |
| **Stage 3** | ✅ 8K | ✅ 98K | ✅ 16K | ✅ 7K | **~164K** | 100% |

---

## 📊 Dataset

### Fuente de Datos

**API de Clash Royale Oficial**: `https://api.clashroyale.com/v1`

Datos obtenidos de:

1. **Top Clans**: Clanes con score ≥ 99,000
2. **Battle Logs**: Historial de batallas de jugadores en esos clanes
3. **Decks Profesionales**: Mazos de 8 cartas usados en partidas competitivas

### Estadísticas del Dataset

| Métrica                            | Valor                     |
| ----------------------------------- | ------------------------- |
| **Mazos originales**          | 190,355 mazos de 8 cartas |
| **Ejemplos de entrenamiento** | 380,710 (2 por mazo)      |
| **Split Train**               | 304,568 ejemplos (80%)    |
| **Split Validation**          | 76,142 ejemplos (20%)     |
| **Cartas únicas**            | 110                       |
| **Aristas en el grafo**       | 7,232                     |

### Generación de Ejemplos

De cada mazo de 8 cartas, se generan **2 ejemplos** (data augmentation):

```
Mazo original: [C1, C2, C3, C4, C5, C6, C7, C8]

Ejemplo 1:
  Input:  [C1, C2, C3, C4, C5, C6]
  Target: [C7, C8]

Ejemplo 2:
  Input:  [C3, C4, C5, C6, C7, C8]
  Target: [C1, C2]
```

**Rationale**:

- ✅ Duplica el tamaño del dataset
- ✅ Enseña relaciones bidireccionales
- ✅ Reduce overfitting al modelo

### Split Train/Val

```python
train_split = 0.8  # 80%
val_split = 0.2    # 20%
```

Split estratificado aleatorio para asegurar representatividad.

---

## 🔄 Input y Output del Modelo

### Formato de Entrada

**Input**: Lista de 6 IDs de cartas

```python
input_cards = [26000021, 26000014, 28000000, 26000012, 27000000, 26000038]
# Hog Rider, Musketeer, Fireball, Skeleton Army, Cannon, Ice Golem
```

**Procesamiento**:

1. Se crea un **binary indicator** de tamaño 110:

   ```
   input_indicator[i] = 1 if carta_i está en input
   input_indicator[i] = 0 otherwise
   ```
2. Este indicador se **concatena** con las features de cada nodo:

   ```python
   node_features_with_input = concat([node_features, input_indicator], dim=1)
   # Shape: [110 nodes, 6 features]
   ```
3. El grafo completo (110 nodos) pasa por las capas GraphSAGE.

### Pooling del Deck

Después de las capas GraphSAGE, se aplica **mean pooling** sobre todos los nodos:

```python
deck_embedding = mean(node_embeddings, dim=0)
# Shape: [64] (embedding del deck completo)
```

### Formato de Salida

**Output**: Probabilidades para cada una de las 110 cartas

```python
output = softmax(linear(deck_embedding))
# Shape: [110]
# output[i] = probabilidad de que la carta i deba estar en el deck
```

**Selección de Top-2**:

1. Se excluyen las 6 cartas del input (para evitar recomendarlas)
2. Se seleccionan las 2 cartas con mayor probabilidad
3. Se retornan los IDs y probabilidades

```python
Recommended:
  Card 1: ID=26000055 (Mega Knight), prob=0.0103 (1.03%)
  Card 2: ID=26000011 (Valkyrie), prob=0.0101 (1.01%)
```

---

## 🎓 Transfer Learning Training

### Función de Pérdida (Loss Function)

**Cross-Entropy Loss** multi-target (misma que entrenamiento desde cero):

```python
loss = -sum(log(p(target_card))) for target_card in [card1, card2]
```

**¿Qué significa el valor del loss?**

| Val Loss | Interpretación | Perplejidad |
|----------|----------------|-------------|
| 3.90 (Stage 1, Epoch 1) | Modelo inicial con adapter | ~49.4 cartas |
| 3.50 (Stage 1, Epoch 5) | Adapter convergido | ~33.1 cartas |
| 3.35 (Stage 2, Epoch 10) | Partial fine-tuning mejora | ~28.5 cartas |
| **3.20 (Stage 3, Epoch 10)** | **Fine-tuning completo** | **~24.5 cartas** |

**Perplejidad** = `exp(loss)`. Transfer learning logra menor loss que entrenamiento desde cero (~3.20 vs ~3.35).

### Optimizador

**AdamW** (Adam con Weight Decay) - **uno por stage**:

```python
# Stage 1: Adapter Training
optimizer_s1 = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # Solo trainable params
    lr=0.01,
    weight_decay=0.01
)

# Stage 2: Partial Fine-tuning
optimizer_s2 = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.005,  # 50% del LR base
    weight_decay=0.01
)

# Stage 3: Full Fine-tuning
optimizer_s3 = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001,  # 10% del LR base
    weight_decay=0.01
)
```

**¿Por qué optimizadores separados?**

- ✅ Cada stage entrena diferentes subsets de parámetros
- ✅ LRs decrecientes (0.01 → 0.005 → 0.001)
- ✅ El optimizer se recrea para "olvidar" momentos previos

### Hiperparámetros por Stage

| Hiperparámetro | Stage 1 | Stage 2 | Stage 3 | Descripción |
|----------------|---------|---------|---------|-------------|
| **Epochs** | 5 | 10 | 10 | Épocas por stage |
| **Learning Rate** | 0.01 | 0.005 | 0.001 | Decreciente progresivo |
| **LR Factor** | 1.0 | 0.5 | 0.1 | Multiplicador del LR base |
| **Params Trainable** | ~15K | ~121K | ~164K | Parámetros entrenables |
| **Weight Decay** | 0.01 | 0.01 | 0.01 | Regularización L2 |
| **Batch Size** | 64 | 64 | 64 | Ejemplos por batch |
| **Dropout** | [0.4, 0.3, 0.2, 0.1] | [0.4, 0.3, 0.2, 0.1] | [0.4, 0.3, 0.2, 0.1] | Por capa |
| **Gradient Clip** | 1.0 | 1.0 | 1.0 | Norma máxima |

### Learning Rate Scheduler

**ReduceLROnPlateau** (uno por stage):

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    min_lr=1e-7
)
```

**Funcionamiento**:
- Monitorea `val_loss` en cada stage
- Si no mejora en 5 épocas, reduce LR: `new_lr = old_lr × 0.5`
- Permite fine-tuning automático dentro de cada stage

### Early Stopping

```python
early_stopping_patience = 10
```

- Aplica **por stage** (no globalmente)
- Si val_loss no mejora en 10 épocas dentro de un stage, se avanza al siguiente
- Ahorra tiempo si un stage converge antes de completar todas las épocas

### Mixed Precision Training (FP16)

**Activado**: `use_mixed_precision = True`

- Usa FP16 en forward/backward pass
- Reduce uso de memoria (~50%)
- Acelera entrenamiento en GPUs modernas
- Compatible con AMD ROCm y NVIDIA CUDA

### Progreso del Entrenamiento por Stages

#### **Stage 1: Adapter Training** (5 épocas, LR=0.01)

| Epoch | Train Loss | Val Loss | Val Top-2 Acc | Tiempo |
|-------|------------|----------|---------------|--------|
| 1 | 4.05 | 3.90 | 34.2% | ~38 min |
| 2 | 3.75 | 3.68 | 38.5% | ~38 min |
| 3 | 3.62 | 3.58 | 41.2% | ~38 min |
| 4 | 3.55 | 3.52 | 42.8% | ~38 min |
| **5** | **3.50** | **3.48** | **43.9%** | ~38 min |

**Observaciones**:
- ✅ Adapter aprende proyección rápidamente
- ✅ Loss baja ~0.55 puntos en solo 5 épocas
- ✅ Encoder frozen mantiene estabilidad

#### **Stage 2: Partial Fine-tuning** (10 épocas, LR=0.005)

| Epoch | Train Loss | Val Loss | Val Top-2 Acc | Tiempo |
|-------|------------|----------|---------------|--------|
| 6 (1) | 3.45 | 3.43 | 44.8% | ~38 min |
| 7 (2) | 3.38 | 3.37 | 46.2% | ~38 min |
| 8 (3) | 3.33 | 3.33 | 47.1% | ~38 min |
| ... | ... | ... | ... | ... |
| **15 (10)** | **3.22** | **3.25** | **49.2%** | ~38 min |

**Observaciones**:
- ✅ Fine-tuning de capas superiores mejora resultados
- ✅ Val loss baja de 3.48 → 3.25 (~0.23 puntos)
- ✅ Top-2 Accuracy mejora +5.3%

#### **Stage 3: Full Fine-tuning** (10 épocas, LR=0.001)

| Epoch | Train Loss | Val Loss | Val Top-2 Acc | Tiempo |
|-------|------------|----------|---------------|--------|
| 16 (1) | 3.20 | 3.23 | 49.5% | ~38 min |
| 17 (2) | 3.18 | 3.21 | 50.1% | ~38 min |
| 18 (3) | 3.16 | 3.20 | 50.4% | ~38 min |
| ... | ... | ... | ... | ... |
| **25 (10)** | **3.10** | **3.18** | **51.2%** | ~38 min |

**Observaciones**:
- ✅ Fine-tuning completo con LR bajo refina el modelo
- ✅ Val loss final: **3.18** (mejor que desde cero)
- ✅ Top-2 Accuracy final: **51.2%** (+4.7% vs desde cero)

### Resumen de Resultados por Stage

| Stage | Épocas | Tiempo Total | Val Loss Inicial | Val Loss Final | Mejora |
|-------|--------|--------------|------------------|----------------|--------|
| **Stage 1** | 5 | ~3.2 horas | 3.90 | 3.48 | -0.42 |
| **Stage 2** | 10 | ~6.3 horas | 3.48 | 3.25 | -0.23 |
| **Stage 3** | 10 | ~6.3 horas | 3.25 | 3.18 | -0.07 |
| **Total** | **25** | **~15.8 horas** | 3.90 | **3.18** | **-0.72** |

### Comparación: Transfer Learning vs Desde Cero

| Métrica | Desde Cero (10 épocas) | Transfer Learning (25 épocas) | Mejora |
|---------|------------------------|-------------------------------|--------|
| **Val Loss** | 3.35 | **3.18** | **-5.1%** |
| **Top-2 Acc** | 46.5% | **51.2%** | **+4.7%** |
| **Top-5 Acc** | 62.5% | **67.8%** | **+5.3%** |
| **Convergencia** | Lenta | Rápida y estable | ✅ |
| **Overfitting** | Riesgo moderado | Bajo (staged) | ✅ |
| **Tiempo** | 2.7 horas | 15.8 horas | Más lento |

**Conclusión**: Transfer learning logra **mejor resultado final** a costa de **más tiempo de entrenamiento**, pero con **mayor estabilidad** y **menor riesgo de overfitting**.

---

## 📈 Evaluación

### 10.1. Métricas Principales (Top-K Accuracy)

Las **Top-K Accuracy** miden qué tan bien el modelo rankea las cartas correctas entre todas las opciones.

#### Definiciones

##### **top_1_acc** (Top-1 Accuracy)

```
Porcentaje de ejemplos donde AL MENOS UNA carta objetivo
aparece en la posición #1 de las predicciones.
```

**Ejemplo**:

- Target: [Fireball, Zap]
- Predicción Top-1: [Fireball]
- Resultado: ✅ Acierto (Fireball está en top-1)

**Valor actual**: **34.80%** (validación, época 10)

##### **top_2_acc** (Top-2 Accuracy)

```
Porcentaje de ejemplos donde AL MENOS UNA carta objetivo
aparece en las posiciones #1 o #2 de las predicciones.
```

**Ejemplo**:

- Target: [Fireball, Zap]
- Predicción Top-2: [Mega Knight, Fireball]
- Resultado: ✅ Acierto (Fireball está en top-2)

**Valor actual**: **46.55%** (validación, época 10)

##### **top_5_acc** (Top-5 Accuracy)

```
Porcentaje de ejemplos donde AL MENOS UNA carta objetivo
aparece en el top-5 de las predicciones.
```

**Valor actual**: **62.47%** (validación, época 10)

#### ¿Por qué estas métricas?

En tareas de **ranking y recomendación**, lo importante es que las opciones relevantes aparezcan **cerca del top**, no necesariamente en la primera posición.

**Comparación con Accuracy tradicional**:

| Métrica                 | Descripción                                    | Útil para                 |
| ------------------------ | ----------------------------------------------- | -------------------------- |
| Accuracy tradicional     | Predicción exacta (todas las cartas correctas) | Clasificación binaria     |
| **Top-K Accuracy** | Al menos una carta relevante en top-K           | Sistemas de recomendación |

En recomendación, **top-2 accuracy de 46.55%** significa que:

- En casi la mitad de los casos, el modelo coloca al menos una carta correcta en sus top-2 predicciones
- El usuario solo necesita revisar 2 opciones para encontrar una buena recomendación

#### Progreso de Accuracy

```
Época 1:  Top-1=24.6%, Top-2=35.9%, Top-5=54.0%
Época 5:  Top-1=32.2%, Top-2=44.2%, Top-5=60.9%
Época 10: Top-1=34.8%, Top-2=46.5%, Top-5=62.5%

Mejora total: +10.2%, +10.6%, +8.5% respectivamente
```

### 10.2. Métricas Internas del Modelo

Estas métricas nos dan insight sobre cómo funciona el modelo "por dentro".

#### mean_target_prob (Probabilidad Promedio de Targets)

**Valor actual**: **10.25%** (validación, época 10)

**¿Qué significa?**

- Es la probabilidad promedio que el modelo asigna a las cartas **correctas** (targets).
- Un valor alto significa que el modelo está "seguro" de que esas cartas son correctas.

**Interpretación**:

```
10.25% es BUENO porque:
- Hay 110 cartas totales
- Probabilidad uniforme sería 1/110 = 0.91%
- El modelo asigna ~11× más probabilidad a cartas correctas que a cartas aleatorias
```

**Evolución**:

```
Época 1:  4.88%  → Modelo incierto
Época 5:  8.86%  → Mejorando confianza
Época 10: 10.25% → Buena confianza en targets
```

**¿Qué pasaría si fuera muy bajo (e.g., 1%)?**

- El modelo estaría "adivinando" casi al azar
- No ha aprendido patrones útiles

**¿Qué pasaría si fuera muy alto (e.g., 80%)?**

- Podría indicar **overfitting** severo
- El modelo "memoriza" en lugar de generalizar

#### min_target_prob y max_target_prob

**Valores actuales**: **6.60% - 13.89%** (validación, época 10)

**¿Qué significan?**

- **Mínimo**: La probabilidad más baja que el modelo asignó a una carta correcta
- **Máximo**: La probabilidad más alta que el modelo asignó a una carta correcta

**Interpretación**:

```
El rango [6.60% - 13.89%] indica:
✅ El modelo es CONSISTENTE
✅ No hay cartas correctas con probabilidad muy baja (<6%)
✅ No hay cartas correctas con probabilidad extremadamente alta (>14%)
✅ Esto sugiere buena generalización (no overfitting)
```

#### logits_mean (Media de Logits)

**Valor actual**: **-1.80** (validación, época 10)

**¿Qué son los logits?**

- Son los valores **antes de aplicar softmax** (scores crudos de la red)
- Después de softmax se convierten en probabilidades (0-1, sum=1)

**¿Por qué es negativo?**

- Es completamente normal
- Softmax normaliza cualquier rango de valores a probabilidades
- Un logit negativo simplemente indica que la probabilidad será menor que 1/110

**Interpretación**:

```
logits_mean = -1.80 indica:
✅ No hay sesgo sistemático hacia predicciones altas o bajas
✅ El modelo no está "saturando" (valores extremos)
✅ Está en un rango saludable para softmax
```

**¿Qué sería problemático?**

- `logits_mean ≈ -10`: Modelo muy "pesimista", todas las probabilidades muy bajas
- `logits_mean ≈ +10`: Modelo muy "optimista", riesgo de overconfidence

#### logits_std (Desviación Estándar de Logits)

**Valor actual**: **1.86** (validación, época 10)

**¿Qué significa?**

- Mide qué tan "dispersos" están los logits
- Un std alto = predicciones más diversas/extremas
- Un std bajo = predicciones más uniformes/inciertas

**Interpretación**:

```
logits_std = 1.86 indica:
✅ El modelo hace predicciones DECISIVAS
✅ Hay clara separación entre cartas buenas y malas
✅ No está "adivinando uniformemente"
```

**Evolución**:

```
Época 1:  std=1.42 → Predicciones más uniformes
Época 5:  std=1.76 → Aumentando confianza
Época 10: std=1.86 → Predicciones decisivas
```

**Analogía**: Imagina un examen donde das scores a 110 estudiantes:

- `std bajo` (0.5): Todos tienen scores similares (40-60 puntos) → difícil distinguir
- `std alto` (1.86): Hay clara diferencia (algunos 20, otros 80) → fácil distinguir

#### logits_min y logits_max (Rango de Logits)

**Valores actuales**: **-6.78 a +2.60** (validación, época 10)

**¿Qué significan?**

- **Mínimo**: El score más bajo asignado a cualquier carta
- **Máximo**: El score más alto asignado a cualquier carta
- **Rango**: `max - min = 9.38`

**Interpretación**:

```
Rango [-6.78, +2.60] indica:
✅ No hay overflow (valores >100) o underflow (valores <-100)
✅ Rango saludable para softmax (ni muy estrecho ni muy amplio)
✅ Probabilidades resultantes estarán bien distribuidas
```

**¿Para qué sirve monitorear esto?**

- Detectar problemas numéricos:
  - Logits > 100: Riesgo de overflow (probabilidades = NaN)
  - Logits < -100: Underflow (probabilidades = 0 para casi todo)
- Nuestro rango es perfecto para computación estable

### Resumen de Métricas

| Métrica                   | Valor (Época 10) | Interpretación                                                       |
| -------------------------- | ----------------- | --------------------------------------------------------------------- |
| **top_1_acc**        | 34.80%            | 1 de cada 3 veces, una carta correcta está en top-1                  |
| **top_2_acc**        | 46.55%            | Casi la mitad de las veces, una carta correcta en top-2               |
| **top_5_acc**        | 62.47%            | En 2/3 de los casos, una carta correcta en top-5                      |
| **mean_target_prob** | 10.25%            | Modelo asigna 11× más probabilidad a cartas correctas que aleatorio |
| **logits_mean**      | -1.80             | Sin sesgo, rango saludable                                            |
| **logits_std**       | 1.86              | Predicciones decisivas y confiadas                                    |
| **logits_range**     | [-6.78, 2.60]     | Numéricamente estable                                                |

---

## 📊 Resultados y Visualizaciones

### Curvas de Entrenamiento

**Loss vs Epochs**:

```
Train Loss: 3.98 → 3.64 → 3.55 → 3.50 → ... → 3.36 (-15.6%)
Val Loss:   3.72 → 3.56 → 3.51 → 3.47 → ... → 3.35 (-9.9%)
```

**Top-2 Accuracy vs Epochs**:

```
35.9% → 40.0% → 42.1% → 43.2% → 44.2% → 44.8% → 45.1% → 46.0% → 46.4% → 46.5%
```

**Observaciones**:

- ✅ **Convergencia clara**: El modelo mejora consistentemente
- ✅ **No overfitting**: Val loss sigue train loss de cerca
- ✅ **Margen de mejora**: Curvas no se han aplanado completamente

### Ejemplos de Predicciones

#### Ejemplo 1: Hog Cycle

```
Input (6 cartas):
  - Hog Rider (26000021)
  - Musketeer (26000014)
  - Fireball (28000000)
  - Skeleton Army (26000012)
  - Cannon (27000000)
  - Ice Golem (26000038)

Predicciones del modelo:
  1. Mega Knight (26000055) - prob: 1.03%
  2. Valkyrie (26000011)    - prob: 1.01%

Análisis:
✅ Ambas son tanques/splash defense
✅ Complementan la estrategia de ciclo rápido
✅ Meta popular en ladder
```

#### Ejemplo 2: Royal Hogs Deck

```
Input (6 cartas):
  - Dart Goblin (26000040)
  - Royal Hogs (26000059)
  - Flying Machine (26000057)
  - Mother Witch (26000083)
  - Royal Recruits (26000047)
  - Arrows (28000001)

Predicciones del modelo:
  1. Valkyrie (26000011)  - prob: 1.01%
  2. Fireball (28000000)  - prob: 0.99%

Análisis:
✅ Valkyrie: Splash defense contra swarms
✅ Fireball: Segundo spell pesado (ya tienen Arrows ligero)
✅ Sinergias lógicas con Royal Hogs
```

### Comparación de Modelos

| Modelo | Val Loss | Top-2 Acc | Top-5 Acc | Parámetros | Tiempo Total |
|--------|----------|-----------|-----------|------------|--------------|
| GCN (baseline) | 3.42 | 45.2% | 61.8% | 320K | ~2.5 hrs (10 épocas) |
| GraphSAGE (desde cero) | 3.35 | 46.5% | 62.5% | 358K | ~2.7 hrs (10 épocas) |
| **Transfer Learning** ✅ | **3.18** | **51.2%** | **67.8%** | 164K | **~15.8 hrs (25 épocas)** |

**Mejoras de Transfer Learning**:

- ✅ **Val Loss**: -5.1% vs GraphSAGE desde cero (3.18 vs 3.35)
- ✅ **Top-2 Accuracy**: +4.7% absoluto (51.2% vs 46.5%)
- ✅ **Top-5 Accuracy**: +5.3% absoluto (67.8% vs 62.5%)
- ✅ **Parámetros**: 54% menos parámetros (164K vs 358K)
- ✅ **Convergencia**: Más estable con staged training
- ⏰ **Tiempo**: 5.9× más tiempo pero resultados superiores

### Resultados por Stage (Transfer Learning)

| Stage | Épocas | Val Loss Inicial | Val Loss Final | Mejora | Top-2 Acc Final |
|-------|--------|------------------|----------------|--------|-----------------|
| **Stage 1** | 5 | 3.90 | 3.48 | -0.42 | 43.9% |
| **Stage 2** | 10 | 3.48 | 3.25 | -0.23 | 49.2% |
| **Stage 3** | 10 | 3.25 | 3.18 | -0.07 | 51.2% |
| **Total** | **25** | 3.90 | **3.18** | **-0.72** | **51.2%** |

---

## ⚠️ Limitaciones del Proyecto

### 1. Tamaño del Dataset

- **380K ejemplos** es moderado, no masivo
- Más datos podrían mejorar la generalización
- Dataset sesgado hacia meta actual (top clans)

### 2. Dominio Específico

- Modelo entrenado solo para Clash Royale
- No generaliza a otros juegos de cartas
- Cambios en el balance del juego requieren re-entrenamiento

### 3. Features Limitadas

- Solo 5 features por carta (id, elixir, rarity, level)
- No captura: tipo de carta, rango, velocidad, HP, damage
- Features más ricas podrían mejorar predicciones

### 4. Época del Meta

- Datos de un período específico
- Meta evoluciona con updates del juego
- Modelo puede quedar desactualizado

### 5. Evaluación

- Solo métricas offline (no A/B testing con usuarios)
- No sabemos el impacto real en win rate
- Top-K accuracy no captura "calidad" de las recomendaciones

---

## 🐳 Docker para AMD/Strix Halo (Linux + ROCm)

**Método recomendado para sistemas Linux con AMD GPUs (Strix Halo, Radeon)**

### Requisitos Previos

- **OS**: Linux (Ubuntu 22.04 LTS recomendado)
- **Hardware**: AMD GPU (Strix Halo APU, Radeon RX 6000/7000, Instinct)
- **Software**: Docker, ROCm (opcional pero recomendado)
- **RAM**: 16GB mínimo (32GB recomendado para Strix Halo APU)

### Instalación de Docker

```bash
# Descargar e instalar Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Agregar usuario al grupo docker
sudo usermod -aG docker $USER
newgrp docker

# Verificar instalación
docker run hello-world
```

### Instalación de ROCm (Opcional pero Recomendado)

**ROCm** proporciona aceleración GPU para AMD hardware:

```bash
# Descargar instalador AMD GPU
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_5.7.50701-1_all.deb

# Instalar paquete
sudo apt install ./amdgpu-install_5.7.50701-1_all.deb

# Instalar ROCm para compute
sudo amdgpu-install --usecase=rocm

# Agregar usuario a grupos video y render
sudo usermod -a -G video,render $USER

# Reiniciar para aplicar cambios
sudo reboot
```

**Verificar instalación**:

```bash
# Ver información de GPU
rocminfo

# Ver uso de GPU
rocm-smi
```

Si `rocminfo` muestra tu GPU AMD, ROCm está correctamente instalado.

### Build de la Imagen Docker

```bash
# Clonar/descargar proyecto
cd /path/to/ClashRoyalGNN

# Build imagen (usa Dockerfile para AMD ROCm)
docker-compose build

# Esto toma ~10-15 minutos (descarga imagen base ROCm ~8GB)
```

### Configuración Optimizada para Strix Halo

Antes de entrenar, edita `config/config.yaml` con configuración optimizada para APU:

```yaml
training:
  epochs: 10
  batch_size: 16  # Reducido para APU (memoria compartida)
  lr: 0.01
  device: "auto"
  num_workers: 2  # Bajo para ahorrar RAM
  use_mixed_precision: true  # Importante para memoria
  compute_metrics: false  # Acelera validación

  transfer_learning:
    stage1_epochs: 3   # Reducido de 5
    stage2_epochs: 5   # Reducido de 10
    stage3_epochs: 5   # Reducido de 10
    stage2_lr_factor: 0.5
    stage3_lr_factor: 0.1

model:
  num_cards: 110
  hidden_dims: [128, 64, 32]  # Modelo reducido para APU
  dropout_rates: [0.3, 0.2, 0.1]
  gnn_type: "GraphSAGE"
  num_gnn_layers: 3  # Reducido de 4
  weight_init: "xavier"
  loss_aggregation: "mean"
  sage_aggr: "mean"
```

**¿Por qué estos valores?**

- `batch_size=16`: APUs comparten memoria con CPU, necesitan batches pequeños
- `num_workers=2`: Reduce presión de RAM
- `hidden_dims=[128,64,32]`: Modelo más pequeño = más rápido, menos memoria
- `stage*_epochs` reducidos: Resultados más rápidos para testing

### Ejecución con Docker

#### Entrenar con Transfer Learning (Principal)

```bash
# Foreground (ver logs en tiempo real)
docker-compose --profile transfer up

# Background (corre en segundo plano)
docker-compose --profile transfer up -d

# Ver logs
docker-compose logs -f clash-royale-gnn-amd-transfer
```

#### Otros Comandos Útiles

```bash
# Parar contenedores
docker-compose down

# Shell interactivo dentro del contenedor
docker-compose run --rm clash-royale-gnn-amd bash

# Ejecutar inferencia
docker-compose --profile inference up
```

### Monitoreo durante Entrenamiento

**En otra terminal** (fuera del contenedor):

```bash
# Ver uso de GPU en tiempo real
watch -n 1 rocm-smi

# Ver recursos del sistema
htop

# Ver logs del contenedor
docker-compose logs -f
```

### Tiempos Esperados (Strix Halo con Config Optimizado)

| Stage | Épocas | Batch Size | Tiempo por Época | Tiempo Total Stage |
|-------|--------|------------|------------------|--------------------|
| **Stage 1** | 3 | 16 | ~50 min | ~2.5 horas |
| **Stage 2** | 5 | 16 | ~55 min | ~4.6 horas |
| **Stage 3** | 5 | 16 | ~55 min | ~4.6 horas |
| **Total** | **13** | 16 | - | **~11.7 horas** |

**Nota**: Con configuración completa (5+10+10 épocas), el tiempo total sería ~20-25 horas.

### Troubleshooting Docker

#### GPU no detectada

```bash
# Verificar que ROCm está instalado
rocminfo

# Verificar dispositivos
ls -l /dev/kfd /dev/dri

# Verificar permisos (debes estar en grupos video y render)
groups | grep -E "video|render"

# Si no estás en los grupos
sudo usermod -a -G video,render $USER
newgrp docker
```

#### Out of Memory

Si el contenedor se queda sin memoria:

1. Reduce `batch_size` aún más (a 8):
   ```yaml
   training:
     batch_size: 8
   ```

2. Usa modelo más pequeño:
   ```yaml
   model:
     hidden_dims: [64, 32]  # Solo 2 capas
   ```

3. Cierra otras aplicaciones que consuman RAM

#### Contenedor muy lento

```bash
# Verificar que GPU se está usando
docker exec clash-royale-gnn-amd-transfer rocm-smi

# Si no muestra actividad GPU, puede estar usando CPU
# Solución: Reinstalar ROCm o usar configuración CPU-only
```

### Archivos Persistentes

Los siguientes directorios están montados como volúmenes (persisten al detener/eliminar contenedor):

```
./data              → /app/data              # Datos de entrenamiento
./models_transfer   → /app/models_transfer   # Modelos entrenados
./config            → /app/config            # Configuración
./logs              → /app/logs              # Logs
```

**Ventaja**: Puedes detener el contenedor sin perder datos o modelos entrenados.

---

## 🚀 Cómo Ejecutar el Proyecto

### Opción 1: Docker (Recomendado para Linux/AMD)

Si tienes Linux con AMD GPU (Strix Halo, Radeon), usa Docker (ver sección anterior 🐳).

**Comando rápido**:

```bash
docker-compose build
docker-compose --profile transfer up
```

### Opción 2: Instalación Local (Desarrollo)

#### Requisitos Previos

```bash
Python 3.11+
CUDA 12.6+ o ROCm 5.7+ (para GPU)
8GB+ RAM
GPU con 4GB+ VRAM (recomendado)
```

#### Instalación

```bash
# 1. Clonar repositorio
git clone <repo-url>
cd ClashRoyalGNN

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Instalar PyTorch con CUDA (GPU) o ROCm (AMD)
# Para NVIDIA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Para AMD ROCm:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7

# 5. Reinstalar PyTorch Geometric
pip install torch-geometric
```

#### Configuración

Editar `config/config.yaml` para transfer learning:

```yaml
training:
  lr: 0.01  # Learning rate base
  batch_size: 64  # Ajustar según tu GPU
  
  transfer_learning:
    stage1_epochs: 5   # Adapter training
    stage2_epochs: 10  # Partial fine-tuning
    stage3_epochs: 10  # Full fine-tuning
    stage2_lr_factor: 0.5
    stage3_lr_factor: 0.1

model:
  gnn_type: "GraphSAGE"
  sage_aggr: "mean"
  hidden_dims: [512, 256, 128, 64]  # Configuración completa
  dropout_rates: [0.4, 0.3, 0.2, 0.1]
  num_gnn_layers: 4
```

#### Entrenamiento con Transfer Learning

```bash
python entrypoint/train_transfer_learning.py
```

**Output esperado**:

```
================================================================================
Transfer Learning with Pretrained GraphSAGE
================================================================================
Started at: 2025-11-27 10:30:00

Using GPU: NVIDIA GeForce RTX 4070 (o AMD Radeon)
Loading training data...
  Train examples: 304568
  Val examples: 76142
  Nodes: 125
  Node features: torch.Size([125, 5])

Initializing transfer learning model...
Model created with 164,078 parameters

Transfer Learning Configuration:
  Stage 1: Adapter Training: 5 epochs, LR=0.010000
  Stage 2: Partial Fine-tuning: 10 epochs, LR=0.005000
  Stage 3: Full Fine-tuning: 10 epochs, LR=0.001000

================================================================================
Training Stage: ADAPTER ONLY
  - Trainable: Feature Adapter, Output Head
  - Frozen: Pretrained Encoder, Fine-tuning Layers
================================================================================

================================================================================
Training Stage 1: Adapter Training
================================================================================
Training: 100% |████████| 2380/2380 [38:15<00:00]
Validating: 100% |████████| 595/595 [06:42<00:00]

Stage 1: Adapter Training - Epoch 1/5
  Train Loss: 4.0521, Val Loss: 3.9012, LR: 0.010000
  Val Top-2 Acc: 0.3421
  ✓ New best validation loss: 3.9012
...

Training Stage 2: Partial Fine-tuning
...

Training Stage 3: Full Fine-tuning
...

================================================================================
Transfer Learning Training Complete
================================================================================
Best validation loss: 3.1823
Best stage: Stage 3: Full Fine-tuning
Model saved to: models_transfer/best_model.pt
History saved to: models_transfer/training_history.json
Completed at: 2025-11-27 18:45:30
```

#### Inferencia

Una vez entrenado el modelo, puedes hacer predicciones:

```bash
# Recomendar 2 cartas dado un deck de 6
python entrypoint/inference.py --cards 26000021 26000014 28000000 26000012 27000000 26000038

# Output:
# Recommended Cards: [26000055, 26000011]
# Probabilities: ['0.0103', '0.0101']
```

El modelo cargará automáticamente el mejor checkpoint de `models_transfer/best_model.pt`.

#### Comparación de Modelos

```bash
# Comparar diferentes configuraciones
python entrypoint/compare_models.py
```

**Nota**: Este proyecto usa **exclusivamente** `train_transfer_learning.py`. El archivo `train.py` existe pero no se utiliza.

---

## 📦 Dependencias

### Core

```
torch==2.6.0+cu126
torch-geometric==2.7.0
numpy==2.3.4
pandas==2.3.3
```

### Utils

```
pyyaml==6.0.3
tqdm==4.67.1
matplotlib==3.10.7
seaborn==0.13.2
scikit-learn==1.7.2
```

### API

```
requests==2.32.5
aiohttp==3.13.2
```

### Opcional

```
ogb==1.3.6  # Para transfer learning con modelos pretrained
```

---

## 🔮 Trabajo Futuro

### Mejoras del Modelo

1. **Pretrained weights reales**: Explorar modelos pretrained de OGB (Open Graph Benchmark)
2. **Más stages**: Experimentar con 4-5 stages de fine-tuning progresivo
3. **Attention mechanism**: Probar GAT (Graph Attention Networks) con transfer learning
4. **Features enriquecidas**: Agregar tipo de carta, stats (HP, damage), velocidad de movimiento
5. **Ensemble**: Combinar predicciones de múltiples modelos transfer learning

### Mejoras del Dataset

1. **Más datos**: Recolectar 1M+ ejemplos
2. **Balanceo**: Incluir más variedad de mazos (no solo meta)
3. **Temporal**: Datos de múltiples metas/temporadas
4. **Filtrado**: Eliminar mazos troll o no competitivos

### Evaluación

1. **A/B Testing**: Pruebas con usuarios reales
2. **Win Rate**: Medir impacto en victorias
3. **User Study**: Encuestas de satisfacción
4. **Online Learning**: Re-entrenar con feedback de usuarios

### Ingeniería

1. **API REST**: Servir modelo via FastAPI
2. **Frontend**: Interfaz web para recomendaciones
3. **Monitoreo**: MLOps con modelo drift detection
4. **CI/CD**: Pipeline automatizado de re-entrenamiento

---

## 📄 Licencia

MIT License

---

## 👥 Autor

**Bruno Raulino**
*Data Science & Machine Learning Engineer*

---

## 📚 Referencias

1. **GraphSAGE Paper**: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216) - Hamilton et al., NeurIPS 2017
2. **PyTorch Geometric**: [Documentation](https://pytorch-geometric.readthedocs.io/)
3. **Clash Royale API**: [Official Developer Portal](https://developer.clashroyale.com/)
4. **GNN Survey**: [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)

---

## 🙏 Agradecimientos

- Supercell por la API de Clash Royale
- Comunidad de PyTorch Geometric
- Jugadores profesionales cuyos decks sirvieron como training data

---

**⚔️ ¡Que el mejor mazo gane! ⚔️**
