using System.Text.Json.Serialization;

namespace NaicsClassifier.Core.Models;

public sealed class BenchmarkCase
{
    [JsonPropertyName("query")]
    public required string Query { get; init; }

    [JsonPropertyName("expected_code")]
    public required string ExpectedCode { get; init; }

    [JsonPropertyName("expected_title")]
    public string ExpectedTitle { get; init; } = string.Empty;

    [JsonPropertyName("category")]
    public string Category { get; init; } = "standard"; // standard, semantic, typo, ambiguous

    [JsonPropertyName("note")]
    public string Note { get; init; } = string.Empty;
}

public sealed class ModelBenchmarkSummary
{
    [JsonPropertyName("model_key")]
    public required string ModelKey { get; init; }

    [JsonPropertyName("model_name")]
    public required string ModelName { get; init; }

    [JsonPropertyName("total_queries")]
    public int TotalQueries { get; init; }

    [JsonPropertyName("top1_hits")]
    public int Top1Hits { get; init; }

    [JsonPropertyName("top1_accuracy")]
    public double Top1Accuracy { get; init; }

    [JsonPropertyName("top3_hits")]
    public int Top3Hits { get; init; }

    [JsonPropertyName("top3_accuracy")]
    public double Top3Accuracy { get; init; }

    [JsonPropertyName("top5_hits")]
    public int Top5Hits { get; init; }

    [JsonPropertyName("top5_accuracy")]
    public double Top5Accuracy { get; init; }

    [JsonPropertyName("average_latency_ms")]
    public double AverageLatencyMs { get; init; }

    [JsonPropertyName("min_latency_ms")]
    public double MinLatencyMs { get; init; }

    [JsonPropertyName("max_latency_ms")]
    public double MaxLatencyMs { get; init; }

    [JsonPropertyName("category_accuracies")]
    public Dictionary<string, double> CategoryAccuracies { get; init; } = [];
}

public sealed class CaseModelResult
{
    [JsonPropertyName("model_key")]
    public required string ModelKey { get; init; }

    [JsonPropertyName("predicted_code")]
    public string PredictedCode { get; init; } = string.Empty;

    [JsonPropertyName("predicted_title")]
    public string PredictedTitle { get; init; } = string.Empty;

    [JsonPropertyName("confidence")]
    public double Confidence { get; init; }

    [JsonPropertyName("score")]
    public double Score { get; init; }

    [JsonPropertyName("is_top1")]
    public bool IsTop1 { get; init; }

    [JsonPropertyName("is_top3")]
    public bool IsTop3 { get; init; }

    [JsonPropertyName("is_top5")]
    public bool IsTop5 { get; init; }

    [JsonPropertyName("elapsed_ms")]
    public double ElapsedMs { get; init; }
}

public sealed class BenchmarkCaseEvaluation
{
    [JsonPropertyName("case_info")]
    public required BenchmarkCase CaseInfo { get; init; }

    [JsonPropertyName("results")]
    public Dictionary<string, CaseModelResult> Results { get; init; } = [];
}

public sealed class BenchmarkSuiteResult
{
    [JsonPropertyName("executed_at")]
    public DateTime ExecutedAt { get; init; } = DateTime.UtcNow;

    [JsonPropertyName("total_cases")]
    public int TotalCases { get; init; }

    [JsonPropertyName("categories")]
    public IReadOnlyList<string> Categories { get; init; } = [];

    [JsonPropertyName("models")]
    public IReadOnlyList<ModelBenchmarkSummary> Models { get; init; } = [];

    [JsonPropertyName("evaluations")]
    public IReadOnlyList<BenchmarkCaseEvaluation> Evaluations { get; init; } = [];
}
