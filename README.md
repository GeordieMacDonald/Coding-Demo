# NAICS Code Classification Engine (.NET 10)

A high-performance, multi-algorithm demonstration platform built in **C# / .NET 10** comparing three foundational Natural Language Processing (NLP) classification approaches against the **North American Industry Classification System (NAICS)**.

The platform accepts natural language business or company descriptions (e.g., *"commercial roofing contractor and siding repair"*, *"litigation advocacy in federal court"*, *"rooffing contrator"* with typos) and predicts the matching **4-digit NAICS Industry Group** (308 categories backed by 20,373 index activities).

---

## 🌟 Key Highlights & Tech Stack

- **Target Framework:** .NET 10 (`net10.0`, C# 13)
- **Algorithms Compared:**
  1. **TF-IDF:** Production pipeline via **Microsoft.ML** (`Microsoft.ML.Transforms.Text`), unigram/bigram token bags, Porter stemming, damped cosine similarity, and keyword boosting.
  2. **FastText:** Supervised character n-gram bag-of-embeddings model (48 dimensions, 60,000 subword hash buckets) accelerated by **.NET 10 SIMD** (`System.Numerics.Tensors.TensorPrimitives`).
  3. **BERT / Transformer:** Dense semantic embeddings via **ONNX Runtime** (`Microsoft.ML.OnnxRuntime`) running an INT8-quantized `all-MiniLM-L6-v2` transformer with contiguous SIMD MaxSim vector scoring against 20,373 official NAICS index terms.
- **Client/Server Architecture:**
  - Modern ASP.NET Core 10 Web API with pre-warmed singleton classifiers.
  - Interactive dark-slate web dashboard supporting **Single-Model**, **Dual-Comparison** (TF-IDF vs FastText), and **Tri-Model Comparison** (TF-IDF vs FastText vs BERT) modes.
  - Built-in **Benchmark Suite** with 22 ground-truth test cases and live browser evaluation scorecard.
  - 46 automated unit and integration tests passing in ~4 seconds.

---

## 🏛 Solution Architecture

```
Internal-Coding-Demo/
├── NaicsClassifier.Core/                # Domain models, repository, interfaces, and benchmark suite
│   ├── Models/
│   │   ├── NaicsRecord.cs              # 4-digit NAICS model with Census hierarchy and index terms
│   │   ├── ClassificationResult.cs     # Multi-candidate predictions, scores, and explanations
│   │   └── BenchmarkModels.cs          # Benchmark cases, evaluation summaries, and metrics
│   ├── Interfaces/
│   │   ├── INaicsClassifier.cs         # Extensible classifier provider contract
│   │   ├── INaicsRepository.cs         # NAICS code/search storage abstraction
│   │   └── IBenchmarkService.cs        # Ground-truth evaluation runner
│   ├── Data/
│   │   ├── EmbeddedNaicsRepository.cs  # In-memory repository with O(1) lookup
│   │   └── naics_2022_4digit.json      # 308 4-digit industry groups & 20,373 index terms
│   └── Services/
│       └── BenchmarkService.cs         # Curated 22-case benchmark suite across 4 test categories
│
├── NaicsClassifier.Engine.TfIdf/        # Microsoft.ML TF-IDF Classifier
│   ├── TfIdfClassifier.cs              # Unigram+bigram bag featurizer, length-damped cosine scoring
│   └── PorterStemmer.cs                # English morphological suffix stemmer
│
├── NaicsClassifier.Engine.FastText/    # FastText Subword Classifier
│   ├── FastTextClassifier.cs           # SIMD TensorPrimitives forward pass, character n-grams (3-5)
│   └── Data/fasttext_naics_4digit.bin  # Precomputed binary weights (48-dim, 60k buckets, 308 classes)
│
├── NaicsClassifier.Engine.Bert/        # Transformer / BERT Semantic Classifier
│   ├── BertClassifier.cs               # ONNX Runtime INT8 all-MiniLM-L6-v2, SIMD MaxSim search
│   ├── WordPieceTokenizer.cs           # Pure C# fast WordPiece tokenizer for uncased vocabulary
│   └── Data/
│       ├── model_quantized.onnx        # INT8 quantized all-MiniLM-L6-v2 ONNX model (22.9 MB)
│       ├── vocab.txt                   # WordPiece vocabulary (231 KB)
│       └── naics_bert_embeddings.bin   # Dense 384-d precomputed vectors for 20,373 index terms (32.5 MB)
│
├── NaicsClassifier.Api/                 # ASP.NET Core 10 Web API & Dashboard
│   ├── Program.cs                      # Minimal API endpoints, DI registration, static files
│   ├── Services/
│   │   ├── ClassifierRegistry.cs       # Pluggable model registry
│   │   └── ClassifierWarmupService.cs  # Background service for zero-lag initial user queries
│   └── wwwroot/
│       └── index.html                  # Single-page dashboard, comparison views, benchmark modal
│
├── NaicsClassifier.Tests/               # Automated Unit, Integration & Benchmark Tests
│   ├── TfIdfClassifierTests.cs         # 14 tests covering TF-IDF accuracy, explainability & speed
│   ├── FastTextClassifierTests.cs      # 8 tests covering FastText SIMD inference & typo resilience
│   ├── BertClassifierTests.cs          # 8 tests covering Transformer inference, synonyms & latency
│   ├── BenchmarkServiceTests.cs        # 2 tests verifying benchmark execution & accuracy bounds
│   └── ApiIntegrationTests.cs          # 14 tests for all HTTP REST API endpoints
│
├── WORKPLAN.md                         # Milestone-by-milestone development tracker
└── README.md                           # This comprehensive guide
```

---

## 🔬 Algorithmic Deep Dive & Trade-Offs

Each of the three algorithms represents a distinct evolutionary generation in Natural Language Processing:

### 1. TF-IDF (`Microsoft.ML`) — Bag-of-Words with Statistical Weighting
- **Mechanism:** Builds a sparse vector representation using term frequency multiplied by inverse document frequency over unigrams and bigrams. Terms are stemmed using a Porter Stemmer to handle morphological inflections (e.g., *roofing* $\to$ *roof*). Similarity is computed via length-damped sparse cosine similarity and query coordination factor.
- **Explainability:** 100% transparent. The engine returns the exact matched terms and feature weights contributing to the score.
- **Latency:** ~0.1 – 1.5 ms on CPU.
- **Memory:** Extremely lightweight (~4 MB).
- **Strengths:** Blazing fast, zero neural overhead, easily auditable in regulated environments.
- **Weaknesses:** Brittle against spelling errors (*"rooffing"* $\to$ 0 matches) and vocabulary mismatch (cannot equate *"oral cavity hygiene"* with *"dentistry"*).

### 2. FastText (`NaicsClassifier.Engine.FastText`) — Character n-Gram Subword Embeddings
- **Mechanism:** Decomposes input words into character n-grams of lengths 3 through 5 (e.g., `<ro`, `roo`, `oof`, `ofi`, `fing`, `ing>`). Subword hashes are mapped into 60,000 buckets and projected into a 48-dimensional dense embedding space. Averaged token vectors are multiplied by class weight matrices using SIMD vector instructions via .NET 10's `System.Numerics.Tensors.TensorPrimitives`.
- **Latency:** ~0.08 – 0.5 ms on CPU.
- **Memory:** ~11 MB.
- **Strengths:** Remarkable resilience against typos, phonetic errors, and novel morphological variants while running faster than TF-IDF.
- **Weaknesses:** Bag-of-ngrams discards sentence structure and word order; struggles with abstract semantic paraphrasing.

### 3. BERT / Transformer (`all-MiniLM-L6-v2` via ONNX Runtime) — Dense Contextual Semantics
- **Mechanism:** Uses a 6-layer INT8-quantized MiniLM Transformer. Queries are tokenized via a high-speed C# WordPiece tokenizer, passed to ONNX Runtime to produce 384-dimensional contextual token vectors, mean-pooled, and $L_2$-normalized. 
- **MaxSim Strategy:** Rather than comparing the query against a single averaged category vector (which dilutes fine-grained signals), the engine computes the maximum cosine similarity (**MaxSim**) against precomputed embeddings of all **20,373 official Census index activities and titles** using SIMD dot products (`TensorPrimitives.Dot`).
- **Latency:** ~4 – 20 ms on CPU.
- **Memory:** ~35 MB (model + precomputed index vectors).
- **Strengths:** True semantic understanding. Accurately maps conceptual synonyms (*"courtroom litigation advocacy"* $\to$ `5411 Legal Services`, *"oral cavity hygiene"* $\to$ `6212 Offices of Dentists`) even when not a single keyword overlaps.
- **Weaknesses:** Higher computational requirements than linear models; neural "black box" representation.

---

## 📊 Live Benchmark Comparison Matrix

Evaluation conducted across the curated 22-case benchmark suite (Standard, Semantic, Typo, and Ambiguous categories) on .NET 10:

| Metric | 📐 TF-IDF (Microsoft.ML) | ⚡ FastText (Subwords) | 🧠 BERT (Transformer INT8) |
|---|---|---|---|
| **Top-1 Accuracy** | 59% (13/22) | 50% (11/22) | **77% (17/22)** |
| **Top-3 Accuracy** | 73% (16/22) | 73% (16/22) | **96% (21/22)** |
| **Top-5 Accuracy** | 77% (17/22) | 77% (17/22) | **100% (22/22)** |
| **Semantic Match Accuracy** | 20% | 0% | **80%** |
| **Typo Resilience Accuracy** | 33% | **50%** | **50%** |
| **Average CPU Latency** | 1.14 ms | **0.09 ms** | 4.51 ms |
| **Throughput (QPS / core)** | ~800 req/sec | **~10,000 req/sec** | ~220 req/sec |
| **RAM Footprint** | **~4 MB** | ~11 MB | ~35 MB |
| **Explainability** | **Exact Tokens & Weights** | Subword Tokens | Nearest Census Index Activity |

### Algorithmic Trade-off Summary:
- **For ultra-high-throughput, latency-critical pipelines with clean data:** Use **TF-IDF** or **FastText**.
- **For user-facing search with noisy/misspelled inputs:** Use **FastText**.
- **For complex natural language descriptions, conversational AI, and paraphrases:** Use **BERT**.
- **Production Recommendation (Hybrid Cascade):** Route query through TF-IDF/FastText first; if top confidence $< 60\%$, fall back to the BERT semantic model.

---

## 🚀 Getting Started

### Prerequisites
- [.NET 10 SDK](https://dotnet.microsoft.com/download/dotnet/10.0) (`10.0.100` or newer)
- Windows, macOS, or Linux (fully cross-platform)

### Build and Run

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Internal-Coding-Demo
   ```

2. **Run all unit & integration tests:**
   ```bash
   dotnet test
   ```
   *(All 46 unit, integration, and benchmark tests will execute in ~3-4 seconds)*

3. **Start the Web API & Dashboard:**
   ```bash
   dotnet run --project NaicsClassifier.Api
   ```

4. **Open the Dashboard:**
   Navigate your browser to:
   ```
   http://localhost:5115
   ```

---

## 🌐 Web Dashboard Features

- **Tri-Model Comparison Mode:** View predictions from TF-IDF, FastText, and BERT side-by-side with live per-model latency, confidence bars, and explanation chips.
- **Interactive Sample Pills:**
  - *Standard Queries:* Custom Software / SaaS, Commercial Roofing, Poultry Hatchery, Italian Restaurant, Law Firm, Plumbing.
  - *Semantic / Synonym Generalization:* Courtroom Litigation, Oral Cavity Hygiene, Pipe Leak & Furnace, Veterinary Clinic.
  - *Typo Tests:* Misspelled roofing, misspelled SaaS, misspelled plumbing, misspelled dining.
- **Benchmark Matrix Modal:** Click **"📊 Benchmark Suite"** in the top navigation bar to run live evaluations across all 22 test cases and inspect the side-by-side results scorecard.
- **Industry Detail Modal:** Click **"Details"** on any prediction card to view official U.S. Census descriptions, Sector/Subsector hierarchy, and all associated 6-digit NAICS index activities.

---

## 📡 REST API Reference

### 1. Classify Text
- **Endpoint:** `POST /api/classify`
- **Request Body:**
  ```json
  {
    "query": "commercial roofing contractor and siding repair",
    "algorithm": "bert",
    "top_k": 3
  }
  ```
  *(Supported algorithms: `tfidf`, `fasttext`, `bert`)*
- **Sample Response:**
  ```json
  {
    "algorithm": "BERT / Transformer (all-MiniLM-L6-v2)",
    "query": "commercial roofing contractor and siding repair",
    "elapsed_ms": 4.12,
    "top_candidates": [
      {
        "code": "2381",
        "title": "Foundation, Structure, and Building Exterior Contractors",
        "score": 0.742,
        "confidence": 0.81,
        "sector_code": "23",
        "sector_title": "Construction",
        "explanation": "Dense semantic similarity 0.742 via BERT contextual embeddings",
        "matched_terms": ["Roofing contractors", "Siding contractors"]
      }
    ]
  }
  ```

### 2. List Registered Models
- **Endpoint:** `GET /api/models`
- **Response:**
  ```json
  [
    { "key": "tfidf", "name": "TF-IDF (Microsoft.ML)", "description": "Statistical bag-of-words..." },
    { "key": "fasttext", "name": "FastText (Subword Embeddings)", "description": "Character n-gram embeddings..." },
    { "key": "bert", "name": "BERT / Transformer (all-MiniLM-L6-v2)", "description": "Quantized INT8 semantic transformer..." }
  ]
  ```

### 3. NAICS Code Details & Hierarchy
- **Endpoint:** `GET /api/naics/{code}` (e.g. `/api/naics/5415`)
- Returns the 4-digit code, title, official Census description, Sector, Subsector, and all associated Census index activities.

### 4. Run Benchmark Suite Live
- **Endpoint:** `POST /api/benchmark/run`
- **Request Body:** `{ "top_k": 5 }`
- Executes all 22 benchmark cases across all 3 algorithms simultaneously and returns model summaries, category accuracies, and case-by-case outcomes.

---

## 🎭 Live Demonstration Walkthrough Script

When presenting this project to teammates or stakeholders, follow this 4-step sequence:

### Step 1: Baseline Exact Match (TF-IDF Excels)
1. In the search box, click **"Roofing Contractor"** (`commercial roofing contractor and siding repair`).
2. Show all three models predict **2381 Foundation, Structure, and Building Exterior Contractors** with high confidence.
3. Highlight that **TF-IDF** executes in ~1 ms and produces exact matched token stems (`roof`, `contractor`, `side`, `repair`).

### Step 2: The Typo Problem (FastText Excels)
1. Under **Typo / Subword Tests**, click **"Typo: Roofing Contractor"** (`commercl rooffing contrator and sidng repaiir`).
2. Observe the results:
   - **TF-IDF:** Misses or drops confidence significantly because word-level vocabulary does not match misspelled tokens.
   - **FastText:** Immediately identifies `2381` at #1 in **0.1 ms** because character n-grams (`roo`, `oof`, `fing`) match the subword vocabulary.
   - **BERT:** Also resilient through WordPiece sub-tokens (`roof`, `##fing`).

### Step 3: Semantic Generalization (BERT Excels)
1. Under **Semantic / BERT Generalization**, click **"Semantic: Courtroom Litigation"** (`litigation advocacy and legal representation in court`).
2. Observe the dramatic difference:
   - **TF-IDF:** Predicts `8133 Social Advocacy Organizations` because of the exact word "advocacy".
   - **FastText:** Also misclassifies under `8133` due to subword alignment on advocacy.
   - **BERT:** Correctly predicts **5411 Legal Services** as #1 (61% confidence), identifying semantic alignment with "Legal aid services" and "Paralegal services".
3. Try **"Semantic: Oral Cavity Hygiene"** (`oral cavity hygiene and tooth cleaning clinic`):
   - Only BERT correctly maps this to **6212 Offices of Dentists** without the word "dentist" appearing in the query!

### Step 4: The Benchmark Scorecard
1. Click **"📊 Benchmark Suite"** in the top navigation bar.
2. Review the live 22-case evaluation table:
   - Point out **FastText's 0.09 ms latency**.
   - Point out **BERT's 96% Top-3 accuracy and 80% semantic match rate**.
   - Show how the three models form a complementary accuracy/speed spectrum.

---

## 📄 License & Attribution
- NAICS 2022 dataset sourced from the [U.S. Census Bureau](https://www.census.gov/naics/).
- Sentence Transformer model `all-MiniLM-L6-v2` by [Sentence-Transformers / Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Built with Microsoft .NET 10, Microsoft.ML, and ONNX Runtime.
