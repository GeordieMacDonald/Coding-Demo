using System.Diagnostics;
using System.Numerics.Tensors;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NaicsClassifier.Core.Models;
using NaicsClassifier.Core.Services;

namespace NaicsClassifier.Engine.Bert;

public sealed class BertClassifier : INaicsClassifier, IDisposable
{
    public string Key => "bert";
    public string Name => "BERT / Transformer (all-MiniLM-L6-v2)";
    public string Description => "Dense semantic embeddings (384-d) via ONNX Runtime with MaxSim contextual semantic matching.";
    public bool IsTrained { get; private set; }

    private WordPieceTokenizer? _tokenizer;
    private InferenceSession? _session;
    private readonly SemaphoreSlim _lock = new(1, 1);

    // Precomputed index vectors:
    // We group vectors by category code for fast MaxSim search
    private sealed class CategoryVectorGroup
    {
        public required NaicsRecord Record { get; init; }
        public required int VectorCount { get; init; }
        public required float[] Vectors { get; init; } // VectorCount * 384
    }

    private CategoryVectorGroup[] _categoryGroups = [];
    private const int EmbeddingDim = 384;

    public async Task InitializeAsync(IReadOnlyList<NaicsRecord> records, CancellationToken cancellationToken = default)
    {
        if (records == null || records.Count == 0)
        {
            throw new ArgumentException("Records collection cannot be null or empty.", nameof(records));
        }

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (IsTrained) return;

            await Task.Run(() =>
            {
                var recordsByCode = records.ToDictionary(r => r.Code);

                // 1. Load Tokenizer
                var assembly = Assembly.GetExecutingAssembly();
                using var vocabStream = GetResourceOrFileStream(assembly, "vocab.txt");
                _tokenizer = new WordPieceTokenizer(vocabStream);

                // 2. Load ONNX Model
                var modelPath = ResolveDataFilePath("model_quantized.onnx");
                var sessionOptions = new SessionOptions
                {
                    IntraOpNumThreads = Math.Min(4, Environment.ProcessorCount),
                    ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };
                _session = new InferenceSession(modelPath, sessionOptions);

                // 3. Load Precomputed Embeddings
                var embeddingsPath = ResolveDataFilePath("naics_bert_embeddings.bin");
                using var fs = File.OpenRead(embeddingsPath);
                using var reader = new BinaryReader(fs);

                int totalVectors = reader.ReadInt32();
                int dim = reader.ReadInt32();
                if (dim != EmbeddingDim)
                {
                    throw new InvalidDataException($"Embedding dimension mismatch: expected {EmbeddingDim}, got {dim}.");
                }

                // Group vectors by 4-digit code
                var rawGroups = new Dictionary<string, List<float[]>>();
                for (int i = 0; i < totalVectors; i++)
                {
                    var codeBytes = reader.ReadBytes(4);
                    var code = System.Text.Encoding.ASCII.GetString(codeBytes).Trim();
                    var vec = new float[EmbeddingDim];
                    for (int d = 0; d < EmbeddingDim; d++)
                    {
                        vec[d] = reader.ReadSingle();
                    }

                    if (!rawGroups.TryGetValue(code, out var list))
                    {
                        list = new List<float[]>();
                        rawGroups[code] = list;
                    }
                    list.Add(vec);
                }

                // Flatten each group into a single contiguous array for cache-friendly SIMD traversal
                var groupsList = new List<CategoryVectorGroup>();
                foreach (var (code, vecs) in rawGroups)
                {
                    if (!recordsByCode.TryGetValue(code, out var record)) continue;

                    int vCount = vecs.Count;
                    var flat = new float[vCount * EmbeddingDim];
                    for (int v = 0; v < vCount; v++)
                    {
                        Buffer.BlockCopy(vecs[v], 0, flat, v * EmbeddingDim * sizeof(float), EmbeddingDim * sizeof(float));
                    }

                    groupsList.Add(new CategoryVectorGroup
                    {
                        Record = record,
                        VectorCount = vCount,
                        Vectors = flat
                    });
                }

                _categoryGroups = groupsList.ToArray();
                IsTrained = true;
            }, cancellationToken);
        }
        finally
        {
            _lock.Release();
        }
    }

    public Task<ClassificationResult> ClassifyAsync(string query, int topK = 5, CancellationToken cancellationToken = default)
    {
        if (!IsTrained || _session == null || _tokenizer == null)
        {
            throw new InvalidOperationException("Classifier is not initialized. Call InitializeAsync first.");
        }

        var sw = Stopwatch.StartNew();

        if (string.IsNullOrWhiteSpace(query))
        {
            return Task.FromResult(new ClassificationResult
            {
                Query = query,
                Algorithm = Name,
                ElapsedMilliseconds = 0,
                TopCandidates = []
            });
        }

        // 1. Tokenize query
        var tokenIds = _tokenizer.Tokenize(query.Trim(), maxTokens: 64);
        int seqLen = tokenIds.Length;

        // 2. Prepare ONNX input tensors
        var inputIds = new DenseTensor<long>(new[] { 1, seqLen });
        var attentionMask = new DenseTensor<long>(new[] { 1, seqLen });
        var tokenTypeIds = new DenseTensor<long>(new[] { 1, seqLen });

        for (int i = 0; i < seqLen; i++)
        {
            inputIds[0, i] = tokenIds[i];
            attentionMask[0, i] = 1;
            tokenTypeIds[0, i] = 0;
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds)
        };

        // 3. Run BERT forward pass
        float[] queryEmbedding = new float[EmbeddingDim];
        using (var results = _session.Run(inputs))
        {
            var outputTensor = results.First(r => r.Name == "last_hidden_state").AsTensor<float>();

            // Mean pooling over tokens (excluding special tokens if preferred, or standard attention mask mean)
            for (int i = 0; i < seqLen; i++)
            {
                for (int d = 0; d < EmbeddingDim; d++)
                {
                    queryEmbedding[d] += outputTensor[0, i, d];
                }
            }
            for (int d = 0; d < EmbeddingDim; d++)
            {
                queryEmbedding[d] /= seqLen;
            }

            // L2 normalize
            float norm = MathF.Sqrt(TensorPrimitives.SumOfSquares<float>(queryEmbedding));
            if (norm > 1e-8f)
            {
                TensorPrimitives.Divide(queryEmbedding, norm, queryEmbedding);
            }
        }

        // 4. MaxSim category scoring with SIMD Dot products
        var scoredCandidates = new (CategoryVectorGroup Group, float MaxScore)[_categoryGroups.Length];

        for (int i = 0; i < _categoryGroups.Length; i++)
        {
            var group = _categoryGroups[i];
            float maxSim = float.NegativeInfinity;

            for (int v = 0; v < group.VectorCount; v++)
            {
                var vecSpan = group.Vectors.AsSpan(v * EmbeddingDim, EmbeddingDim);
                float sim = TensorPrimitives.Dot(queryEmbedding.AsSpan(), vecSpan);
                if (sim > maxSim)
                {
                    maxSim = sim;
                }
            }

            scoredCandidates[i] = (group, maxSim);
        }

        // 5. Sort top candidates
        Array.Sort(scoredCandidates, (a, b) => b.MaxScore.CompareTo(a.MaxScore));
        int returnCount = Math.Min(topK, scoredCandidates.Length);

        // 6. Calibrate confidence scores using softmax over top-10 candidate similarities
        int poolCount = Math.Min(10, scoredCandidates.Length);
        float temperature = 0.06f; // Fine temperature scale for cosine similarities in [0.4, 0.8] range
        float maxScore = scoredCandidates[0].MaxScore;

        float sumExp = 0f;
        float[] expScores = new float[poolCount];
        for (int i = 0; i < poolCount; i++)
        {
            float exp = MathF.Exp((scoredCandidates[i].MaxScore - maxScore) / temperature);
            expScores[i] = exp;
            sumExp += exp;
        }

        var candidates = new List<CandidateMatch>(returnCount);
        for (int i = 0; i < returnCount; i++)
        {
            var (group, rawScore) = scoredCandidates[i];
            float confidence = sumExp > 0 ? (expScores[i] / sumExp) : 0f;

            // Extract matching terms: find which index terms from this category match query words
            var matchedTerms = ExtractRelevantTerms(query, group.Record);

            candidates.Add(new CandidateMatch
            {
                Code = group.Record.Code,
                Title = group.Record.Title,
                SectorCode = group.Record.SectorCode,
                SectorTitle = group.Record.SectorTitle,
                SubsectorCode = group.Record.SubsectorCode,
                SubsectorTitle = group.Record.SubsectorTitle,
                Score = Math.Round(rawScore, 4),
                Confidence = Math.Round(confidence, 4),
                Explanation = $"Dense semantic similarity {rawScore:F3} via BERT contextual embeddings",
                MatchedTerms = matchedTerms
            });
        }

        sw.Stop();

        return Task.FromResult(new ClassificationResult
        {
            Query = query,
            Algorithm = Name,
            ElapsedMilliseconds = Math.Round(sw.Elapsed.TotalMilliseconds, 2),
            TopCandidates = candidates
        });
    }

    private static IReadOnlyList<string> ExtractRelevantTerms(string query, NaicsRecord record)
    {
        var queryWords = query.ToLowerInvariant()
            .Split(new[] { ' ', ',', '.', ';', '-', '/', '&' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(w => w.Length > 2)
            .ToHashSet();

        var matches = new List<string>();
        foreach (var term in record.IndexTerms)
        {
            var termLower = term.ToLowerInvariant();
            if (queryWords.Any(w => termLower.Contains(w)))
            {
                matches.Add(term);
                if (matches.Count >= 3) break;
            }
        }

        if (matches.Count == 0 && record.IndexTerms.Count > 0)
        {
            matches.Add(record.IndexTerms[0]);
        }

        return matches;
    }

    private static string ResolveDataFilePath(string filename)
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "Data", filename),
            Path.Combine(AppContext.BaseDirectory, filename),
            Path.Combine(Directory.GetCurrentDirectory(), "NaicsClassifier.Engine.Bert", "Data", filename),
            Path.Combine(Directory.GetCurrentDirectory(), "Data", filename),
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "Data", filename)
        };

        foreach (var path in candidates)
        {
            if (File.Exists(path)) return Path.GetFullPath(path);
        }

        throw new FileNotFoundException($"BERT data file '{filename}' could not be located in output directories.");
    }

    private static Stream GetResourceOrFileStream(Assembly assembly, string filename)
    {
        var resName = assembly.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(filename, StringComparison.OrdinalIgnoreCase));
        if (resName != null)
        {
            return assembly.GetManifestResourceStream(resName)!;
        }

        var filePath = ResolveDataFilePath(filename);
        return File.OpenRead(filePath);
    }

    public void Dispose()
    {
        _session?.Dispose();
        _lock.Dispose();
    }
}
