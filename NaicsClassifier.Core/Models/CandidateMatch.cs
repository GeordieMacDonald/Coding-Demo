using System.Text.Json.Serialization;

namespace NaicsClassifier.Core.Models;

public sealed class CandidateMatch
{
    [JsonPropertyName("code")]
    public required string Code { get; init; }

    [JsonPropertyName("title")]
    public required string Title { get; init; }

    [JsonPropertyName("sector_code")]
    public string SectorCode { get; init; } = string.Empty;

    [JsonPropertyName("sector_title")]
    public string SectorTitle { get; init; } = string.Empty;

    [JsonPropertyName("subsector_code")]
    public string SubsectorCode { get; init; } = string.Empty;

    [JsonPropertyName("subsector_title")]
    public string SubsectorTitle { get; init; } = string.Empty;

    [JsonPropertyName("score")]
    public double Score { get; init; }

    [JsonPropertyName("confidence")]
    public double Confidence { get; init; }

    [JsonPropertyName("explanation")]
    public string Explanation { get; init; } = string.Empty;

    [JsonPropertyName("matched_terms")]
    public IReadOnlyList<string> MatchedTerms { get; init; } = [];
}
