using System.Text.Json.Serialization;

namespace NaicsClassifier.Core.Models;

public sealed class ClassificationRequest
{
    [JsonPropertyName("text")]
    public required string Text { get; init; }

    [JsonPropertyName("algorithm")]
    public string? Algorithm { get; init; }

    [JsonPropertyName("top_k")]
    public int TopK { get; init; } = 5;

    [JsonPropertyName("min_score")]
    public double MinScore { get; init; } = 0.0;
}
