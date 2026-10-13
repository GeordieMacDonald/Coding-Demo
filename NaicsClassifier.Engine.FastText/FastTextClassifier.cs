using System.Diagnostics;
using System.Numerics.Tensors;
using System.Reflection;
using System.Text.RegularExpressions;
using NaicsClassifier.Core.Models;
using NaicsClassifier.Core.Services;

namespace NaicsClassifier.Engine.FastText;

public sealed partial class FastTextClassifier : INaicsClassifier
{
    public string Key => "fasttext";
    public string Name => "FastText (Subword Embeddings)";
    public string Description => "Supervised subword character n-gram bag-of-embeddings model, resilient to typos, misspellings, and morphological variants.";
    public bool IsTrained { get; private set; }

    private int _bucketSize;
    private int _embedDim;
    private int _numClasses;
    private float[] _embeddings = [];
    private float[] _fcWeights = [];
    private float[] _fcBias = [];
    private NaicsRecord[] _classRecords = [];
    private readonly SemaphoreSlim _lock = new(1, 1);

    [GeneratedRegex("[a-zA-Z0-9]+")]
    private static partial Regex WordRegex();

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
                var assembly = Assembly.GetExecutingAssembly();
                const string resourceName = "NaicsClassifier.Engine.FastText.Data.fasttext_naics_4digit.bin";

                using var stream = assembly.GetManifestResourceStream(resourceName)
                    ?? throw new InvalidOperationException($"Embedded resource '{resourceName}' not found in {assembly.FullName}.");

                using var reader = new BinaryReader(stream);

                // Read magic
                var magic = new string(reader.ReadChars(4));
                if (magic != "FTXT")
                {
                    throw new InvalidDataException($"Invalid FastText model format: header was '{magic}', expected 'FTXT'.");
                }

                _bucketSize = reader.ReadInt32();
                _embedDim = reader.ReadInt32();
                _numClasses = reader.ReadInt32();

                int embCount = _bucketSize * _embedDim;
                _embeddings = new float[embCount];
                var embBytes = reader.ReadBytes(embCount * sizeof(float));
                Buffer.BlockCopy(embBytes, 0, _embeddings, 0, embBytes.Length);

                int fcWeightCount = _numClasses * _embedDim;
                _fcWeights = new float[fcWeightCount];
                var fcBytes = reader.ReadBytes(fcWeightCount * sizeof(float));
                Buffer.BlockCopy(fcBytes, 0, _fcWeights, 0, fcBytes.Length);

                _fcBias = new float[_numClasses];
                var biasBytes = reader.ReadBytes(_numClasses * sizeof(float));
                Buffer.BlockCopy(biasBytes, 0, _fcBias, 0, biasBytes.Length);

                // Match classes 0..numClasses-1 to sorted NAICS records
                _classRecords = records.OrderBy(r => r.Code).ToArray();
                if (_classRecords.Length != _numClasses)
                {
                    throw new InvalidOperationException($"Class count mismatch: model has {_numClasses} classes, records have {_classRecords.Length}.");
                }

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
        if (!IsTrained)
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

        // 1. Extract subword features and word tokens
        var (features, extractedWords) = ExtractFeatures(query);
        if (features.Count == 0)
        {
            features.Add(0);
        }

        // 2. Average embeddings across all extracted n-gram feature hashes
        float[] textVector = new float[_embedDim];
        float scale = 1.0f / features.Count;

        foreach (int bucketId in features)
        {
            int offset = bucketId * _embedDim;
            var embSpan = _embeddings.AsSpan(offset, _embedDim);
            TensorPrimitives.Add(textVector, embSpan, textVector);
        }
        TensorPrimitives.Multiply(textVector, scale, textVector);

        // 3. Compute logits: fcWeights * textVector + fcBias
        float[] logits = new float[_numClasses];
        for (int c = 0; c < _numClasses; c++)
        {
            int fcOffset = c * _embedDim;
            var weightSpan = _fcWeights.AsSpan(fcOffset, _embedDim);
            float dot = TensorPrimitives.Dot(weightSpan, textVector);
            logits[c] = dot + _fcBias[c];
        }

        // 4. Softmax probabilities
        float maxLogit = TensorPrimitives.Max(logits);
        float[] expValues = new float[_numClasses];
        float sumExp = 0f;

        for (int c = 0; c < _numClasses; c++)
        {
            float exp = MathF.Exp(logits[c] - maxLogit);
            expValues[c] = exp;
            sumExp += exp;
        }

        float invSum = sumExp > 0f ? (1.0f / sumExp) : 1f;
        float[] probabilities = new float[_numClasses];
        TensorPrimitives.Multiply(expValues, invSum, probabilities);

        // 5. Rank top-k
        var topIndices = Enumerable.Range(0, _numClasses)
            .OrderByDescending(i => probabilities[i])
            .Take(topK)
            .ToList();

        var candidates = new List<CandidateMatch>(topIndices.Count);
        foreach (int idx in topIndices)
        {
            var record = _classRecords[idx];
            float prob = probabilities[idx];
            float rawLogit = logits[idx];

            // Subword explanation: matched words and character n-grams
            var matchedKeywords = extractedWords
                .Where(w => record.AggregatedContent.Contains(w, StringComparison.OrdinalIgnoreCase))
                .Take(5)
                .ToList();

            string explanation = matchedKeywords.Count > 0
                ? $"Subword & n-gram alignment on '{string.Join("', '", matchedKeywords)}' (subword embeddings)"
                : "Matched via subword character n-gram embedding similarity (handles typos/OOV).";

            candidates.Add(new CandidateMatch
            {
                Code = record.Code,
                Title = record.Title,
                SectorCode = record.SectorCode,
                SectorTitle = record.SectorTitle,
                SubsectorCode = record.SubsectorCode,
                SubsectorTitle = record.SubsectorTitle,
                Score = Math.Round(rawLogit, 4),
                Confidence = Math.Round(Math.Clamp((double)prob, 0.0, 1.0), 4),
                Explanation = explanation,
                MatchedTerms = matchedKeywords.Count > 0 ? matchedKeywords : extractedWords.Take(4).ToList()
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

    private (List<int> Features, List<string> Words) ExtractFeatures(string text)
    {
        var matches = WordRegex().Matches(text.ToLowerInvariant());
        var features = new List<int>(matches.Count * 8);
        var words = new List<string>(matches.Count);

        foreach (Match match in matches)
        {
            string w = match.Value;
            words.Add(w);
            string wrapped = $"<{w}>";

            // Add full word hash
            features.Add(HashString(wrapped));

            // Add character n-grams (3, 4, 5)
            int len = wrapped.Length;
            for (int n = 3; n <= 5; n++)
            {
                if (len >= n)
                {
                    for (int i = 0; i <= len - n; i++)
                    {
                        features.Add(HashString(wrapped.Substring(i, n)));
                    }
                }
            }
        }

        return (features, words);
    }

    private int HashString(string s)
    {
        // 32-bit FNV-1a hash matching Python training implementation
        uint hash = 2166136261u;
        var bytes = System.Text.Encoding.UTF8.GetBytes(s);
        foreach (byte b in bytes)
        {
            hash = (hash ^ b) * 16777619u;
        }
        return (int)(hash % (uint)_bucketSize);
    }
}
