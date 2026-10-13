using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using NaicsClassifier.Core.Models;
using NaicsClassifier.Core.Services;

namespace NaicsClassifier.Engine.TfIdf;

public sealed class TfIdfClassifier : INaicsClassifier
{
    public string Key => "tfidf";
    public string Name => "TF-IDF (Microsoft.ML)";
    public string Description => "Term Frequency - Inverse Document Frequency with unigrams/bigrams and cosine similarity powered by Microsoft.ML.";
    public bool IsTrained { get; private set; }

    private readonly MLContext _mlContext;
    private ITransformer? _model;
    private PredictionEngine<DocumentInput, DocumentFeatures>? _predictionEngine;
    private string[] _vocabulary = [];
    private DocumentVector[] _documentVectors = [];
    private readonly SemaphoreSlim _lock = new(1, 1);

    public TfIdfClassifier(int? seed = 42)
    {
        _mlContext = new MLContext(seed);
    }

    private sealed class DocumentInput
    {
        public string Text { get; set; } = string.Empty;
    }

    private sealed class DocumentFeatures
    {
        public VBuffer<float> Features { get; set; }
    }

    private sealed class DocumentVector
    {
        public required NaicsRecord Record { get; init; }
        public required int[] Indices { get; init; }
        public required float[] Values { get; init; }
        public required float Norm { get; init; }
    }

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
                var trainingData = records.Select(r => new DocumentInput
                {
                    Text = PorterStemmer.StemText(r.AggregatedContent)
                }).ToList();

                var dataView = _mlContext.Data.LoadFromEnumerable(trainingData);

                // Build Microsoft.ML TF-IDF pipeline:
                // 1. Normalize text (lowercase, remove punctuation, diacritics)
                // 2. Tokenize into words
                // 3. Remove English stop words
                // 4. Produce TF-IDF Word Bags (unigrams and bigrams)
                var pipeline = _mlContext.Transforms.Text.NormalizeText("NormalizedText", nameof(DocumentInput.Text))
                    .Append(_mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "NormalizedText"))
                    .Append(_mlContext.Transforms.Text.RemoveDefaultStopWords("CleanTokens", "Tokens"))
                    .Append(_mlContext.Transforms.Text.ProduceWordBags(
                        "Features",
                        "CleanTokens",
                        ngramLength: 2,
                        useAllLengths: true,
                        weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf));

                _model = pipeline.Fit(dataView);
                var transformedData = _model.Transform(dataView);

                // Extract vocabulary slot names for explainability
                var featuresColumn = transformedData.Schema["Features"];
                var vectorType = (VectorDataViewType)featuresColumn.Type;
                var slotNames = default(VBuffer<ReadOnlyMemory<char>>);
                featuresColumn.GetSlotNames(ref slotNames);

                _vocabulary = new string[vectorType.Size];
                if (slotNames.Length > 0)
                {
                    var slotValues = slotNames.GetValues();
                    var slotIndices = slotNames.GetIndices();

                    if (slotNames.IsDense)
                    {
                        for (int i = 0; i < slotValues.Length; i++)
                        {
                            _vocabulary[i] = slotValues[i].ToString();
                        }
                    }
                    else
                    {
                        for (int i = 0; i < slotIndices.Length; i++)
                        {
                            _vocabulary[slotIndices[i]] = slotValues[i].ToString();
                        }
                    }
                }

                _predictionEngine = _mlContext.Model.CreatePredictionEngine<DocumentInput, DocumentFeatures>(_model);

                // Precompute sparse document vectors and norms for fast cosine similarity scoring
                var vectors = new List<DocumentVector>(records.Count);
                for (int i = 0; i < records.Count; i++)
                {
                    var rec = records[i];
                    var prediction = _predictionEngine.Predict(new DocumentInput { Text = trainingData[i].Text });
                    var feat = prediction.Features;

                    var rawValues = feat.GetValues().ToArray();
                    var rawIndices = feat.IsDense
                        ? Enumerable.Range(0, rawValues.Length).ToArray()
                        : feat.GetIndices().ToArray();

                    // Filter out zero entries
                    var nonZeroPairs = rawIndices
                        .Zip(rawValues, (idx, val) => (idx, val))
                        .Where(p => Math.Abs(p.val) > 1e-6f)
                        .OrderBy(p => p.idx)
                        .ToList();

                    var indices = nonZeroPairs.Select(p => p.idx).ToArray();
                    var values = nonZeroPairs.Select(p => p.val).ToArray();

                    float sumSq = 0f;
                    foreach (var val in values)
                    {
                        sumSq += val * val;
                    }
                    float norm = MathF.Sqrt(sumSq);

                    vectors.Add(new DocumentVector
                    {
                        Record = rec,
                        Indices = indices,
                        Values = values,
                        Norm = norm > 0f ? norm : 1f
                    });
                }

                _documentVectors = vectors.ToArray();
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
        if (!IsTrained || _predictionEngine == null)
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

        // Vectorize query with Porter Stemmer
        var stemmedQuery = PorterStemmer.StemText(query);
        DocumentFeatures queryPrediction;
        lock (_predictionEngine)
        {
            queryPrediction = _predictionEngine.Predict(new DocumentInput { Text = stemmedQuery });
        }

        var qFeatures = queryPrediction.Features;
        var qRawValues = qFeatures.GetValues().ToArray();
        var qRawIndices = qFeatures.IsDense
            ? Enumerable.Range(0, qRawValues.Length).ToArray()
            : qFeatures.GetIndices().ToArray();

        var queryTerms = new Dictionary<int, float>();
        float qSumSq = 0f;
        for (int i = 0; i < qRawValues.Length; i++)
        {
            var val = qRawValues[i];
            if (Math.Abs(val) > 1e-6f)
            {
                var idx = qRawIndices[i];
                queryTerms[idx] = val;
                qSumSq += val * val;
            }
        }

        float qNorm = MathF.Sqrt(qSumSq);
        if (qNorm <= 1e-7f || queryTerms.Count == 0)
        {
            sw.Stop();
            return Task.FromResult(new ClassificationResult
            {
                Query = query,
                Algorithm = Name,
                ElapsedMilliseconds = sw.Elapsed.TotalMilliseconds,
                TopCandidates = []
            });
        }

        // Score against all document vectors
        var scoredList = new List<(DocumentVector Doc, float FinalScore, float RawSim, List<string> MatchedTerms)>(_documentVectors.Length);

        foreach (var doc in _documentVectors)
        {
            float dotProduct = 0f;
            List<string>? matched = null;
            int matchedQueryTermsCount = 0;

            for (int i = 0; i < doc.Indices.Length; i++)
            {
                int docIdx = doc.Indices[i];
                if (queryTerms.TryGetValue(docIdx, out float qVal))
                {
                    matchedQueryTermsCount++;
                    dotProduct += doc.Values[i] * qVal;
                    matched ??= [];
                    if (docIdx < _vocabulary.Length && !string.IsNullOrEmpty(_vocabulary[docIdx]))
                    {
                        matched.Add(_vocabulary[docIdx]);
                    }
                }
            }

            if (dotProduct > 1e-6f)
            {
                float rawSim = dotProduct / (doc.Norm * qNorm);
                
                // Damped document length norm prevents lengthy categories with many index terms
                // from being disproportionately penalized compared to tiny categories
                float dampedDocNorm = MathF.Pow(doc.Norm, 0.65f);
                float baseScore = dotProduct / (dampedDocNorm * qNorm);

                // Coordination factor: proportion of unique query terms present in this document
                float coverage = (float)matchedQueryTermsCount / queryTerms.Count;
                float coordBonus = 0.4f + (0.6f * coverage);

                // Title bonus: if any matched term appears directly in the official industry title
                bool titleHit = false;
                if (matched != null)
                {
                    var stemmedTitle = PorterStemmer.StemText(doc.Record.Title);
                    foreach (var m in matched)
                    {
                        if (stemmedTitle.Contains(m, StringComparison.OrdinalIgnoreCase) ||
                            doc.Record.Title.Contains(m, StringComparison.OrdinalIgnoreCase))
                        {
                            titleHit = true;
                            break;
                        }
                    }
                }
                float titleBonus = titleHit ? 1.4f : 1.0f;

                float finalScore = baseScore * coordBonus * titleBonus;
                scoredList.Add((doc, finalScore, Math.Clamp(rawSim, 0f, 1f), matched ?? []));
            }
        }

        // Rank top-k
        var topScored = scoredList
            .OrderByDescending(s => s.FinalScore)
            .Take(topK)
            .ToList();

        // Calculate calibrated confidence scores
        // If highest score is strong, confidence reflects relative margin
        float totalScore = topScored.Sum(s => s.FinalScore);
        var candidates = new List<CandidateMatch>(topScored.Count);

        foreach (var item in topScored)
        {
            // Relative confidence among top matches
            double confidence = totalScore > 0 ? (item.FinalScore / totalScore) : 0.0;
            
            // Format matched terms cleanly
            var distinctMatches = item.MatchedTerms.Distinct().Take(6).ToList();
            var explanation = distinctMatches.Count > 0
                ? $"Matched terms: {string.Join(", ", distinctMatches)}"
                : "Matched based on general textual similarity.";

            candidates.Add(new CandidateMatch
            {
                Code = item.Doc.Record.Code,
                Title = item.Doc.Record.Title,
                SectorCode = item.Doc.Record.SectorCode,
                SectorTitle = item.Doc.Record.SectorTitle,
                SubsectorCode = item.Doc.Record.SubsectorCode,
                SubsectorTitle = item.Doc.Record.SubsectorTitle,
                Score = Math.Round(item.FinalScore, 4),
                Confidence = Math.Round(Math.Clamp(confidence, 0.0, 1.0), 4),
                Explanation = explanation,
                MatchedTerms = distinctMatches
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
}
