using System.Text.Json.Serialization;

namespace NaicsClassifier.Core.Models;

public sealed class ClassificationResult
{
    [JsonPropertyName("query")]
    public required string Query { get; init; }

    [JsonPropertyName("algorithm")]
    public required string Algorithm { get; init; }

    [JsonPropertyName("elapsed_ms")]
    public double ElapsedMilliseconds { get; init; }

    [JsonPropertyName("top_candidates")]
    public IReadOnlyList<CandidateMatch> TopCandidates { get; init; } = [];
}
