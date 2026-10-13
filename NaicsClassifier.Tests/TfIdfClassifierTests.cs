using NaicsClassifier.Core.Services;
using NaicsClassifier.Engine.TfIdf;
using Xunit;

namespace NaicsClassifier.Tests;

public class TfIdfClassifierTests : IAsyncLifetime
{
    private readonly EmbeddedNaicsRepository _repository = new();
    private readonly TfIdfClassifier _classifier = new();

    public async Task InitializeAsync()
    {
        var records = await _repository.GetAllAsync();
        await _classifier.InitializeAsync(records);
    }

    public Task DisposeAsync() => Task.CompletedTask;

    [Fact]
    public void Classifier_ShouldBeTrained()
    {
        Assert.True(_classifier.IsTrained);
        Assert.Equal("tfidf", _classifier.Key);
        Assert.Contains("Microsoft.ML", _classifier.Name);
    }

    [Theory]
    [InlineData("Computer programming and IT systems consulting", "5415")]
    [InlineData("Roofing contractor and siding installation", "2381")]
    [InlineData("Poultry hatchery and chicken egg farming", "1123")]
    [InlineData("Dentist office and dental clinic", "6212")]
    [InlineData("Restaurant and food dining", "7225")]
    [InlineData("Law office and corporate attorney", "5411")]
    [InlineData("Plumbing and HVAC heating contractor", "2382")]
    public async Task ClassifyAsync_ShouldIdentifyCorrectTop4DigitNaicsCode(string query, string expectedCode)
    {
        var result = await _classifier.ClassifyAsync(query, topK: 5);

        Assert.NotNull(result);
        Assert.NotEmpty(result.TopCandidates);
        Assert.True(result.ElapsedMilliseconds >= 0);

        var top = result.TopCandidates[0];
        Assert.Equal(expectedCode, top.Code);
        Assert.True(top.Score > 0, "Score should be positive.");
        Assert.True(top.Confidence > 0, "Confidence should be positive.");
        Assert.NotEmpty(top.MatchedTerms);
        Assert.False(string.IsNullOrWhiteSpace(top.Explanation));
    }

    [Fact]
    public async Task ClassifyAsync_EmptyQuery_ShouldReturnEmptyCandidates()
    {
        var result = await _classifier.ClassifyAsync("    ", topK: 5);

        Assert.NotNull(result);
        Assert.Empty(result.TopCandidates);
    }

    [Fact]
    public async Task ClassifyAsync_Latency_ShouldBeFast()
    {
        // Warm up
        await _classifier.ClassifyAsync("coffee shop cafe", topK: 5);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await _classifier.ClassifyAsync("commercial real estate leasing agency", topK: 5);
        sw.Stop();

        Assert.NotEmpty(result.TopCandidates);
        // Classification should take less than 25ms on in-memory vectorized index
        Assert.True(sw.ElapsedMilliseconds < 50, $"Elapsed was {sw.ElapsedMilliseconds}ms");
    }
}
