# Clash Royale Graph Neural Network (GNN) Card Recommendation System

## 🎯 Objective

Build a Graph Neural Network-based system that recommends 2 additional cards given 6 input cards in Clash Royale, trained on battle data from top players. The system learns card synergies and co-occurrence patterns from real gameplay data to make intelligent deck completion recommendations.

## 🧠 Why Graph Neural Networks?

### The Problem
Traditional recommendation systems treat cards as independent items, ignoring the complex relationships and synergies between them. In Clash Royale, card combinations matter more than individual cards.

### The Solution: GNNs
**Graph Neural Networks** are ideal for this problem because:

1. **Natural Graph Structure**: Cards naturally form a graph where:
   - **Nodes** = Individual cards (110+ cards)
   - **Edges** = Co-occurrence/synergy relationships (cards that appear together in winning decks)
   - **Node Features** = Card properties (elixir cost, rarity, level, etc.)

2. **Message Passing**: GNNs learn by propagating information through the graph, allowing cards to "communicate" with their neighbors. This captures:
   - Direct synergies (cards that work well together)
   - Indirect relationships (cards that share common partners)
   - Contextual recommendations based on the input deck

3. **Flexible Input**: Unlike sequence models, GNNs can handle variable input sizes and learn from the entire graph structure simultaneously.

### Architecture Choice: GCN vs GAT
- **GCN (Graph Convolutional Network)**: Simpler, faster, good for learning local neighborhood patterns
- **GAT (Graph Attention Network)**: More expressive, learns attention weights for neighbors, better for complex relationships
- **Default**: GCN for efficiency, but GAT can be enabled in config for better accuracy

## 📊 System Architecture

### Data Flow Pipeline

```
API Data Collection
    ↓
Raw Battle Logs & Decks
    ↓
Co-occurrence Matrix (Card Synergies)
    ↓
Graph Construction (Nodes + Edges)
    ↓
Feature Engineering (Node Features + Training Examples)
    ↓
GNN Training (Learn Card Relationships)
    ↓
Inference (Recommend 2 Cards Given 6 Input Cards)
```

### Graph Construction

**Nodes (Cards)**:
- Each of the 110+ Clash Royale cards becomes a node
- Node features include: `id`, `elixirCost`, `rarity`, `maxLevel`, `maxEvolutionLevel`
- Binary indicator added during training: `1` if card is in input deck, `0` otherwise

**Edges (Synergies)**:
- Built from co-occurrence matrix of cards appearing together in decks
- Edge weight = frequency of co-occurrence
- Threshold filter: Only edges with `co-occurrence >= edge_threshold` (default: 5)
- **Rationale**: Filters noise, keeps only meaningful synergies from top players

**Training Examples**:
- From each 8-card deck, create 2 examples:
  - Example 1: First 6 cards → Last 2 cards
  - Example 2: Last 6 cards → First 2 cards
- This doubles the training data and teaches the model bidirectional relationships

## 🏗️ Project Structure

```
ClashRoyalGNN/
├── config/
│   └── config.yaml          # Configuration (API token, model params, paths)
├── data/
│   ├── 01-raw/              # Raw API data (cards, clans, battle logs, decks)
│   ├── 02-preprocessed/     # Co-occurrence matrix
│   ├── 03-features/         # Graph structure + training examples
│   └── 04-predictions/      # Model predictions
├── entrypoint/
│   ├── collect_data.py      # Data collection pipeline
│   ├── train.py             # Model training
│   └── inference.py         # Card recommendation
├── src/
│   ├── data_collection/
│   │   ├── cr_api.py        # Clash Royale API client with rate limiting
│   │   └── data_fetcher.py  # Data fetching & processing
│   ├── models/
│   │   └── gnn_model.py     # GNN architecture (GCN/GAT)
│   ├── pipelines/
│   │   ├── feature_eng_pipeline.py  # Graph construction & feature extraction
│   │   ├── training_pipeline.py      # Training loop with GPU optimization
│   │   └── inference_pipeline.py     # Prediction pipeline
│   └── utils.py             # Utilities (config loading, device detection)
├── process_features.py      # Feature engineering script
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (tested with 3.11)
- **Clash Royale API Token** ([Get one here](https://developer.clashroyale.com))
- **CUDA-capable GPU** (recommended, RTX 4070+ for optimal performance)
  - GPU significantly speeds up training (3-6x faster than CPU)
  - System auto-detects GPU, falls back to CPU if unavailable

### Step 1: Clone and Setup

```bash
# Clone repository
git clone <repository-url>
cd ClashRoyalGNN

# Create virtual environment with Python 3.11
python3.11 -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
# Install basic dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU training)
# Check your CUDA version first: nvidia-smi
# For CUDA 12.6 (most recent):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Reinstall torch-geometric to match PyTorch version
pip install torch-geometric

# Verify GPU is available
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**Expected output with GPU**: `CUDA available: True`, `GPU: NVIDIA GeForce RTX 4070...`

### Step 3: Configure API Token

1. Get your API token from [developer.clashroyale.com](https://developer.clashroyale.com)
2. Open `config/config.yaml`
3. Replace the `token` value:

```yaml
api:
  token: "YOUR_API_TOKEN_HERE"
  base_url: "https://api.clashroyale.com/v1"
  rate_limit:
    requests_per_second: 0.8  # Adjust based on your API tier
    max_retries: 3
    retry_delay: 60
```

**Rate Limiting Rationale**:
- Prevents API blocking (429 errors)
- Silver tier: 0.5 req/s (2s between requests)
- Gold tier: 1 req/s
- Legendary tier: 2 req/s
- System automatically retries on 429 with exponential backoff

## 📥 Step 4: Collect Data

The data collection pipeline fetches real battle data from top players to learn card synergies.

### Run Data Collection

```bash
python entrypoint/collect_data.py
```

### What It Does (Step by Step)

#### 4.1 Fetch Cards (`/cards` endpoint)
- **Purpose**: Get all card metadata (ID, name, elixir cost, rarity, etc.)
- **Output**: `data/01-raw/cards.json`
- **Time**: ~1 second
- **Rationale**: Need card metadata for node features and ID mapping

#### 4.2 Fetch Top Clans (`/clans?minScore={score}`)
- **Purpose**: Find high-performing clans (default: `min_clan_score >= 99000`)
- **Output**: `data/01-raw/clans.json`
- **Time**: ~1 second
- **Rationale**: Top clans contain skilled players with optimal deck compositions

#### 4.3 Extract Player Tags (`/clans/{clanTag}/members`)
- **Purpose**: Get player tags from clan members
- **Method**: Calls `/clans/{clanTag}/members` for each clan
- **Output**: List of player tags
- **Time**: ~50 seconds for 60 clans (1 req/s rate limit)
- **Rationale**: Need player tags to fetch their battle logs

#### 4.4 Fetch Battle Logs (`/players/{tag}/battlelog`)
- **Purpose**: Get recent battle history with deck compositions
- **Output**: `data/01-raw/battle_logs.json`
- **Time**: ~38 minutes for 2300 players (1 req/s)
- **Rationale**: Battle logs contain the actual 8-card decks used by top players
- **Incremental Mode**: Use `--incremental` flag to add data without losing existing data

#### 4.5 Extract Decks
- **Purpose**: Parse 8-card decks from battle logs
- **Output**: `data/01-raw/decks.json`
- **Format**: `[{"cards": [id1, id2, ..., id8], "player_tag": "..."}, ...]`
- **Rationale**: Each deck represents a valid, tested card combination

#### 4.6 Build Co-occurrence Matrix
- **Purpose**: Count how often card pairs appear together in decks
- **Output**: `data/02-preprocessed/co_occurrence_matrix.json`
- **Method**: For each deck, count all card pairs (28 pairs per 8-card deck)
- **Filter**: Only pairs with `count >= edge_threshold` (default: 5)
- **Rationale**: 
  - Captures card synergies from real gameplay
  - Threshold filters noise (random co-occurrences)
  - Forms the edge structure of our graph

### Data Collection Options

```bash
# Basic collection
python entrypoint/collect_data.py

# Incremental mode (add data without losing existing)
python entrypoint/collect_data.py --incremental

# Limit number of clans/players for testing
python entrypoint/collect_data.py --max-clans 10 --max-players 100

# See all options
python entrypoint/collect_data.py --help
```

**Expected Output**:
```
Cards: 121 items
Clans processed: 60
Players processed: 2300
Battle logs: 97500
Decks extracted: 336606
Co-occurrence edges: 7232
```

**Time Estimate** (with 1 req/s rate limit):
- 60 clans with members: ~1 minute
- 2300 players battle logs: ~38 minutes
- **Total: ~40-45 minutes** for full collection

## 🔧 Step 5: Process Features (Feature Engineering)

This step transforms raw data into a graph structure ready for GNN training.

### Run Feature Engineering

```bash
python process_features.py
```

### What It Does (Step by Step)

#### 5.1 Extract Node Features
- **Input**: `data/01-raw/cards.json`
- **Process**: Extract features for each card:
  - `id`: Card ID (normalized)
  - `elixirCost`: Elixir cost (1-7)
  - `rarity`: Encoded as integer (COMMON=0, RARE=1, EPIC=2, LEGENDARY=3)
  - `maxLevel`: Maximum level
  - `maxEvolutionLevel`: Evolution level
- **Output**: Node feature matrix `[num_cards, feature_dim]`
- **Rationale**: Node features help the GNN distinguish between cards and learn card-specific patterns

#### 5.2 Build Graph Structure
- **Input**: `data/02-preprocessed/co_occurrence_matrix.json`
- **Process**:
  1. Load co-occurrence counts
  2. Filter by `edge_threshold` (default: 5)
  3. Create edge index `[2, num_edges]` and edge attributes (weights)
- **Output**: Graph connectivity with weighted edges
- **Rationale**: 
  - Edges represent learned synergies from top players
  - Weights indicate strength of relationship
  - Threshold ensures only meaningful relationships are included

#### 5.3 Create Training Examples
- **Input**: `data/01-raw/decks.json`
- **Process**: For each 8-card deck:
  - Example 1: First 6 cards → Last 2 cards
  - Example 2: Last 6 cards → First 2 cards
- **Output**: `data/03-features/training_examples.json`
- **Rationale**:
  - Doubles training data
  - Teaches bidirectional relationships
  - Each example: `{"input_cards": [6 IDs], "target_cards": [2 IDs]}`

#### 5.4 Save Graph Data
- **Output**: `data/03-features/graph_data.json`
- **Contains**:
  - `node_features`: Feature matrix
  - `edge_index`: Graph connectivity
  - `edge_attr`: Edge weights
  - `id_to_index`, `index_to_id`: Mappings
  - `num_nodes`: Number of cards

**Expected Output**:
```
Graph created with 125 nodes
Created 380710 training examples
Features saved to: data/03-features
```

## 🎓 Step 6: Train the GNN Model

Train the model to learn card relationships and synergies from the graph structure.

### Run Training

```bash
python entrypoint/train.py
```

### Training Process Explained

#### 6.1 Data Loading
- **Load**: Graph structure + training examples
- **Split**: 80% train, 20% validation (configurable via `val_split`)
- **Batch Creation**: Uses `Batch.from_data_list()` to create batched graphs
  - **Rationale**: Process multiple examples in parallel on GPU
  - **Implementation**: Removes dict attributes before batching, adds them back after
  - **Batch Size**: 128 examples per batch (configurable)

#### 6.2 GPU Optimization
The training pipeline is optimized for maximum GPU utilization:

**Batching Strategy**:
- **Real Batching**: Uses PyTorch Geometric's `Batch.from_data_list()`
- **Batch Size**: 128 (from config, not hardcoded)
- **Rationale**: Process multiple graphs simultaneously on GPU
- **Implementation**: Each batch contains 128 separate graph instances with shared structure

**Data Loading Optimization**:
- **num_workers**: 4 parallel workers for data loading
- **pin_memory**: True (faster CPU→GPU transfer)
- **persistent_workers**: True (avoids worker recreation overhead)
- **Rationale**: Keep GPU fed with data, minimize idle time

**Expected GPU Utilization**:
- **Before optimization**: ~39% utilization, ~60 it/s
- **After optimization**: 70-90% utilization, 200-400 it/s (3-6x faster)

#### 6.3 Model Architecture
```
Input: Graph with node features + input card indicator
    ↓
GCN/GAT Layers (3 layers: 256 → 128 → 64)
    ↓
Dropout (0.3, 0.2, 0.1) for regularization
    ↓
Output Layer: [num_nodes, num_cards] logits
    ↓
Loss: Cross-entropy on target cards (excluding input cards)
```

**Forward Pass**:
1. Concatenate node features with binary input indicator
2. Pass through GNN layers (message passing)
3. Each node gets updated representation based on neighbors
4. Output layer predicts card probabilities

**Loss Function**:
- Mask out input cards (can't recommend cards already in deck)
- Multi-label loss: Both target cards should be predicted
- Cross-entropy with softmax

#### 6.4 Training Loop
- **Optimizer**: Adam with learning rate 0.001
- **Early Stopping**: Stops if validation loss doesn't improve for 10 epochs
- **Model Saving**: Saves best model based on validation loss
- **Device**: Auto-detects GPU, falls back to CPU

**Expected Output**:
```
Using GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Train examples: 304568, Val examples: 76142
Starting training for 100 epochs...

Epoch 1/100
Training: 100%|████████| 2380/2380 [12:40<00:00, 3.13it/s]
Validating: 100%|████████| 595/595 [01:50<00:00, 5.39it/s]
Train Loss: 18.2460, Val Loss: 18.2270
Saved best model (val_loss: 18.2270)
```

**Training Parameters** (configurable in `config/config.yaml`):
```yaml
training:
  epochs: 100
  batch_size: 128          # Examples per batch
  lr: 0.001                # Learning rate
  device: "auto"           # Auto-detect GPU
  early_stopping_patience: 10
  val_split: 0.2           # 20% validation
  num_workers: 4           # Data loading workers
```

**Output**: `models/best_model.pt` - Trained model checkpoint

## 🎯 Step 7: Test Inference

Recommend 2 cards given 6 input cards using the trained model.

### Run Inference

```bash
# Command line
python entrypoint/inference.py --cards <id1> <id2> <id3> <id4> <id5> <id6>

# Example with actual card IDs
python entrypoint/inference.py --cards 26000000 26000001 26000002 26000003 26000004 26000005
```

### Inference Process

1. **Load Model**: Load trained GNN from `models/best_model.pt`
2. **Create Graph**: Load graph structure, mark input cards
3. **Forward Pass**: Model predicts probabilities for all cards
4. **Filter**: Exclude input cards, get top 2 recommendations
5. **Output**: Recommended card IDs + probabilities

**Expected Output**:
```
============================================================
Clash Royale GNN Card Recommendation - Inference
============================================================
Loading model...
Model loaded from models/best_model.pt
Using GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Loading graph data...
Predicting cards for input: [26000000, 26000001, ...]
Recommended cards: [26000006, 26000007]
Probabilities: [0.2345, 0.1892]
============================================================
```

### Programmatic Usage

```python
from src.pipelines.inference_pipeline import run_inference

result = run_inference(
    input_card_ids=[26000000, 26000001, 26000002, 26000003, 26000004, 26000005]
)

print(f"Input: {result['input_cards']}")
print(f"Recommended: {result['recommended_cards']}")
print(f"Probabilities: {result['probabilities']}")
```

## ⚙️ Configuration Reference

### Complete `config/config.yaml` Structure

```yaml
api:
  token: "YOUR_TOKEN"
  base_url: "https://api.clashroyale.com/v1"
  rate_limit:
    requests_per_second: 0.8
    max_retries: 3
    retry_delay: 60

data:
  raw_dir: "data/01-raw"
  processed_dir: "data/02-preprocessed"
  features_dir: "data/03-features"
  predictions_dir: "data/04-predictions"
  min_clan_score: 99000  # Filter top clans

training:
  epochs: 100
  batch_size: 128         # GPU-optimized batch size
  lr: 0.001
  device: "auto"         # Auto-detect GPU
  early_stopping_patience: 10
  val_split: 0.2
  save_best_model: true
  model_save_dir: "models"
  num_workers: 4         # Data loading workers

model:
  num_cards: 110
  hidden_dims: [256, 128, 64]  # GNN layer dimensions
  dropout_rates: [0.3, 0.2, 0.1]
  gnn_type: "GCN"        # "GCN" or "GAT"
  num_gnn_layers: 3

graph:
  graph_type: "co_occurrence"
  edge_threshold: 5       # Min co-occurrence for edges
  node_features: ["id", "elixirCost", "rarity", "maxLevel", "maxEvolutionLevel"]
```

### Key Parameters Explained

**`batch_size: 128`**:
- Number of training examples processed simultaneously
- Larger = faster training but more GPU memory
- 128 is optimal for RTX 4070 (8GB VRAM)

**`edge_threshold: 5`**:
- Minimum times card pairs must co-occur to form an edge
- Higher = fewer edges, stronger relationships
- Lower = more edges, includes weaker synergies

**`num_workers: 4`**:
- Parallel data loading workers
- Higher = faster data loading, more CPU usage
- On Windows, may need `num_workers: 0` if multiprocessing issues occur

**`gnn_type: "GCN"`**:
- GCN: Faster, good for local patterns
- GAT: Slower but more expressive, learns attention weights

## 🔍 Technical Deep Dive

### Why This Architecture Works

#### 1. Graph Structure Captures Synergies
- **Co-occurrence edges** represent real synergies from top players
- **Message passing** allows cards to learn from their neighbors
- **Node features** help distinguish card types and properties

#### 2. Binary Input Indicator
- Adding `input_cards` as a feature tells the model which cards are already selected
- Model learns to recommend complementary cards, not duplicates
- Loss function explicitly masks input cards

#### 3. Multi-Label Learning
- Predicts 2 target cards simultaneously
- Learns that certain card pairs work well together
- More realistic than single-card prediction

#### 4. GPU Optimization Strategy
- **Batching**: Process 128 graphs in parallel
- **Batch.from_data_list()**: Creates disjoint union of graphs
- **Separate processing**: Each graph's logits processed individually
- **Averaged loss**: Loss averaged across batch for stable gradients

### Data Pipeline Rationale

**Incremental Collection**:
- Allows adding data without re-fetching
- Tracks processed players to avoid duplicates
- Rebuilds co-occurrence matrix with new data
- Essential for long-term data collection

**Co-occurrence Matrix**:
- Counts all card pairs in all decks
- Filters by threshold to remove noise
- Forms the "ground truth" of card relationships
- More data = better edge structure = better recommendations

## 🐛 Troubleshooting

### API Issues

**403 Forbidden Error**:
- Token expired or invalid
- Check token at [developer.clashroyale.com](https://developer.clashroyale.com)
- Verify token in `config/config.yaml`
- Run `python test_token.py` to diagnose

**429 Too Many Requests**:
- Reduce `requests_per_second` in config
- System auto-retries with `retry_delay`
- Check your API tier limits

### GPU Issues

**GPU Not Detected**:
```bash
# Verify CUDA installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install torch-geometric
```

**Low GPU Utilization**:
- Check `batch_size` is 128 (not 1)
- Verify `num_workers > 0` for data loading
- Monitor with `nvidia-smi` during training
- Should see 70-90% utilization with optimized settings

**Out of Memory (OOM)**:
- Reduce `batch_size` to 64 or 32
- Reduce `num_workers` to 2
- Close other GPU applications

### Training Issues

**Loss Not Decreasing**:
- Check learning rate (try 1e-4 or 1e-5)
- Verify training examples are valid
- Check graph has sufficient edges
- Reduce `edge_threshold` to include more relationships

**Windows Multiprocessing Error**:
- Set `num_workers: 0` in config
- Ensure `if __name__ == "__main__"` guard exists (it does)
- Training will be slower but will work

**Insufficient Data**:
- Collect more data: increase `max_players` in `collect_data.py`
- Reduce `edge_threshold` to include more edges
- Check `data/03-features/training_examples.json` has >10k examples

### Data Collection Issues

**Slow Collection**:
- Normal: ~40 minutes for full collection with rate limiting
- Use `--max-clans` and `--max-players` for testing
- Incremental mode allows resuming

**Missing Data**:
- Check API token is valid
- Verify network connection
- Check rate limiting isn't too aggressive
- Review error messages in console

## 📈 Expected Performance

### Training Metrics
- **GPU Utilization**: 70-90% (RTX 4070)
- **Training Speed**: 200-400 it/s (with batch_size=128)
- **Memory Usage**: 1-3GB VRAM
- **Time per Epoch**: ~15 minutes (304k training examples, batch_size=128)
- **Total Training**: ~2-3 hours for 100 epochs (with early stopping)

### Model Performance
- **Validation Loss**: Should decrease over epochs
- **Early Stopping**: Typically triggers after 10-30 epochs
- **Best Model**: Saved when validation loss is lowest

## 🔄 Complete Workflow Summary

```bash
# 1. Setup
python3.11 -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. Configure
# Edit config/config.yaml with your API token

# 3. Collect Data (40-45 minutes)
python entrypoint/collect_data.py

# 4. Process Features (2-5 minutes)
python process_features.py

# 5. Train Model (2-3 hours with GPU)
python entrypoint/train.py

# 6. Test Inference
python entrypoint/inference.py --cards <id1> <id2> <id3> <id4> <id5> <id6>
```

## 📚 Key Files Reference

| File | Purpose | When to Modify |
|------|---------|----------------|
| `config/config.yaml` | All configuration | Before each run |
| `entrypoint/collect_data.py` | Data collection | To adjust collection params |
| `process_features.py` | Feature engineering | Rarely |
| `entrypoint/train.py` | Training entry point | Rarely |
| `src/pipelines/training_pipeline.py` | Training logic | For advanced optimization |
| `src/models/gnn_model.py` | Model architecture | To change GNN type/layers |

## 🎓 Learning Resources

- **PyTorch Geometric**: [Documentation](https://pytorch-geometric.readthedocs.io/)
- **GNNs Explained**: [Graph Neural Networks](https://distill.pub/2021/gnn-intro/)
- **Clash Royale API**: [Developer Portal](https://developer.clashroyale.com)

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]
