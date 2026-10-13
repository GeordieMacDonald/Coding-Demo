# NAICS Code Classification Engine - Work Plan

## 1. Project Overview & Objectives
This project is an internal demonstration platform comparing multiple natural language classification algorithms against the North American Industry Classification System (NAICS). The workflow allows a user to input a business or activity description (e.g., *"commercial roofing contractor"*, *"cloud-based accounting SaaS"*, *"coffee shop roasting own beans"*) and returns top predicted NAICS codes with confidence scores and explanations.

- **Initial Scope Focus:** **4-digit NAICS codes (Industry Groups)**, balancing fine-grained industry specificity with clean classification boundaries (approx. 312 4-digit industry group categories).
- **Technology Base:** **.NET 10 (C# 13/14)** across all services and libraries.
- **Algorithm Roadmap:** 
  1. **TF-IDF:** Implemented using standard, production-grade .NET ML libraries (**Microsoft.ML** text featurization / TF-IDF transform pipeline).
  2. **FastText:** Supervised subword embedding model.
  3. **BERT / Transformer:** Dense semantic embeddings via ONNX Runtime in .NET.

---

## 2. Technical Architecture (.NET 10 Stack)

- **Runtime & SDK:** .NET 10.0 (`net10.0`)
- **Backend / Server:**
  - ASP.NET Core 10 Web API.
  - Extensible provider pattern (`INaicsClassifier`) allowing plug-and-play swapping and side-by-side comparison of classification algorithms.
  - In-memory vector store & caching for low-latency lookups.
- **TF-IDF Engine:**
  - **Microsoft.ML** (`Microsoft.ML.Transforms.Text`), utilizing standard `ProduceWordBags` / `FeaturizeText` with TF-IDF weighting and n-gram extraction, paired with cosine similarity vector scoring.
- **Frontend / Client:**
  - Responsive web client (interactive single-page dashboard) for fast query entry, top-$K$ candidate visualization, confidence scoring, and algorithm comparison.
- **Data Layer:**
  - Official 2022 NAICS dataset filtered and aggregated to **4-digit Industry Groups** (including rollup of descriptions and 6-digit index entries for rich keyword coverage).

---

## 3. Phased Implementation Roadmap

### Phase 1: Project Scaffolding & 4-Digit NAICS Dataset Ingestion [COMPLETED]
- **1.1 Solution Setup (.NET 10):**
  - Create .NET 10 solution (`NaicsClassifier.slnx`):
    - `NaicsClassifier.Core`: Domain models (`NaicsRecord`, `ClassificationResult`, `CandidateMatch`), interfaces (`INaicsClassifier`, `INaicsRepository`).
    - `NaicsClassifier.Engine.TfIdf`: Implementation using **Microsoft.ML** TF-IDF text featurizer.
    - `NaicsClassifier.Api`: REST API endpoints and dependency injection setup.
    - `NaicsClassifier.Tests`: Unit and integration tests.
- **1.2 4-Digit NAICS Data Ingestion:**
  - Official NAICS 2022 dataset ingested: 308 4-digit Industry Groups with 20,373 aggregated index terms.
  - In-memory embedded repository with O(1) code lookup.
- **Milestone Check-in 1:** Dataset structure, code hierarchy, and repository verified.

---

### Phase 2: TF-IDF Algorithm Implementation (Microsoft.ML Pipeline) [COMPLETED]
- **2.1 Standard .NET TF-IDF Pipeline:**
  - Used `Microsoft.ML.Transforms.Text` (`ProduceWordBags` with `WeightingCriteria.TfIdf`, unigrams + bigrams).
  - Integrated `PorterStemmer` for English morphological inflection handling.
- **2.2 Corpus Vectorization & Indexing:**
  - Featurized 308 4-digit categories.
  - Length-damped sparse cosine similarity and query coordination scoring.
- **2.3 Query Inference & Ranking:**
  - Real-time sparse vector transformation, top-K ranking, confidence normalization, and matched term extraction.
- **2.4 Validation & Unit Tests:**
  - 14 automated unit tests verifying diverse industry domains and sub-millisecond execution.
- **Milestone Check-in 2:** TF-IDF accuracy, latency, and sample classifications approved.

---

### Phase 3: Client/Server Integration & Interactive UI [COMPLETED]
- **3.1 API Endpoints:**
  - `POST /api/classify`: Multi-algorithm prediction endpoint.
  - `GET /api/naics/{code}`: Hierarchy and detail lookup.
  - `GET /api/models`: Model registry status.
  - `GET /api/naics/search`: Live keyword search.
- **3.2 Interactive UI:**
  - Modern dark-slate dashboard with sample query pills, confidence bars, and full NAICS modal.
  - Pre-warmed background service for zero-lag initial requests.
- **Milestone Check-in 3:** Client/server TF-IDF classification workflow verified live.

---

### Phase 4: FastText Classifier Implementation [COMPLETED]
- **4.1 FastText Model Integration:**
  - Supervised subword character n-gram bag-of-embeddings model (embedding dim 48, bucket size 60,000, 308 classes).
  - High-performance C# forward pass in .NET 10 via `System.Numerics.Tensors.TensorPrimitives` (SIMD acceleration).
  - Compact pre-trained binary weights embedded in `NaicsClassifier.Engine.FastText`.
- **4.2 Side-by-Side Comparison:**
  - Split-screen comparison view in the UI (TF-IDF vs FastText).
  - Typo demonstration pills illustrating FastText's robustness against severe misspellings where word-level TF-IDF fails.
- **Milestone Check-in 4:** Review FastText implementation, accuracy delta vs. TF-IDF, and typo tolerance.

---

### Phase 5: Transformer / BERT-based Semantic Model [COMPLETED]
- **5.1 Embedding / Sentence-Transformer Engine:**
  - Integrated `all-MiniLM-L6-v2` dense semantic transformer model (384-dimensional embeddings) using `Microsoft.ML.OnnxRuntime` (1.31.0) and `System.Numerics.Tensors` in .NET 10.
  - Implemented `WordPieceTokenizer` for sub-millisecond BERT tokenization.
  - Precomputed contextual dense embeddings across all 308 categories and 20,373 NAICS index terms (`naics_bert_embeddings.bin`, 32 MB).
  - Implemented cache-friendly MaxSim SIMD cosine similarity search executing in ~15 ms on CPU.
- **5.2 Tri-Model Evaluation Matrix:**
  - Created 3-way side-by-side comparison matrix in the web client (TF-IDF vs FastText vs BERT).
  - Added semantic generalization quick pills demonstrating BERT's ability to accurately classify conceptual synonyms (e.g., "litigation advocacy", "oral cavity hygiene") where token-matching and subword models fail.
  - 13 automated unit & integration tests verifying BERT inference, accuracy, and sub-50ms latency (44 total solution tests passing).
- **Milestone Check-in 5:** Review BERT performance, comparison dashboard, and trade-off analysis.

---

### Phase 6: Demonstration Polish & Deliverables [COMPLETED]
- **6.1 Benchmark Suite & Evaluation Matrix:**
  - Automated ground-truth evaluation suite (`BenchmarkService`) comprising 22 curated industry test cases spanning four distinct query types: Standard, Semantic Generalization, Typo / Misspellings, and Ambiguous Multi-Activity queries.
  - Performance scorecard tracking Top-1, Top-3, Top-5 accuracy, category-level accuracy, and latency percentiles.
  - REST endpoints: `GET /api/benchmark/cases` and `POST /api/benchmark/run`.
  - Automated test coverage in `BenchmarkServiceTests.cs` (46 solution tests passing).
- **6.2 Interactive Dashboard Polish:**
  - Added dedicated **"📊 Benchmark Suite"** modal directly into the web UI featuring live one-click evaluation, comparative performance tables, architectural trade-off guides, and clickable per-case drilldown testing.
  - Tri-Model comparison mode set as default demonstration view with real-time per-model latency tracking.
- **6.3 Comprehensive Documentation:**
  - Authored complete `README.md` containing architecture diagrams, algorithmic deep dives, benchmark evaluation results, build/run instructions, and a live demonstration walkthrough script.
- **Final Check-in:** Final walkthrough and handoff.

