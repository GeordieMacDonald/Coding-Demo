using System.Text.Json.Serialization;

namespace NaicsClassifier.Core.Models;

public sealed class NaicsRecord
{
    [JsonPropertyName("code")]
    public required string Code { get; init; }

    [JsonPropertyName("title")]
    public required string Title { get; init; }

    [JsonPropertyName("description")]
    public string Description { get; init; } = string.Empty;

    [JsonPropertyName("sector_code")]
    public string SectorCode { get; init; } = string.Empty;

    [JsonPropertyName("sector_title")]
    public string SectorTitle { get; init; } = string.Empty;

    [JsonPropertyName("subsector_code")]
    public string SubsectorCode { get; init; } = string.Empty;

    [JsonPropertyName("subsector_title")]
    public string SubsectorTitle { get; init; } = string.Empty;

    [JsonPropertyName("index_terms")]
    public IReadOnlyList<string> IndexTerms { get; init; } = [];

    [JsonPropertyName("child_6digit_codes")]
    public IReadOnlyList<string> Child6DigitCodes { get; init; } = [];

    [JsonPropertyName("aggregated_content")]
    public string AggregatedContent { get; init; } = string.Empty;
}
