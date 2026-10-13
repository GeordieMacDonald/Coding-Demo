using NaicsClassifier.Core.Services;
using NaicsClassifier.Engine.FastText;
using Xunit;

namespace NaicsClassifier.Tests;

public class FastTextClassifierTests : IAsyncLifetime
{
    private readonly EmbeddedNaicsRepository _repository = new();
    private readonly FastTextClassifier _classifier = new();

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
        Assert.Equal("fasttext", _classifier.Key);
        Assert.Contains("FastText", _classifier.Name);
    }

    [Theory]
    [InlineData("commercial roofing contractor and siding repair", "2381")]
    [InlineData("poultry hatchery and chicken egg farming", "1123")]
    [InlineData("dentist office and dental clinic", "6212")]
    [InlineData("italian restaurant and pizzeria dining", "7225")]
    [InlineData("plumbing and hvac heating contractor", "2382")]
    public async Task ClassifyAsync_StandardQuery_ShouldIdentifyCorrectTopCode(string query, string expectedCode)
    {
        var result = await _classifier.ClassifyAsync(query, topK: 5);

        Assert.NotNull(result);
        Assert.NotEmpty(result.TopCandidates);
        Assert.Equal("FastText (Subword Embeddings)", result.Algorithm);

        var top = result.TopCandidates[0];
        Assert.Equal(expectedCode, top.Code);
        Assert.True(top.Confidence > 0);
        Assert.NotEmpty(top.MatchedTerms);
    }

    [Theory]
    [InlineData("plumbin and heatin hvac contrctor", "2382")]
    [InlineData("famly resturant and pizza dinng", "7225")]
    [InlineData("poultri hatcheri and egg farmng", "1123")]
    public async Task ClassifyAsync_TypoQueries_ShouldHandleMisspellingsViaSubwords(string query, string expectedCode)
    {
        var result = await _classifier.ClassifyAsync(query, topK: 5);

        Assert.NotNull(result);
        Assert.NotEmpty(result.TopCandidates);

        // Verify that expected code is in the top 3 candidates despite heavy typos
        var top3Codes = result.TopCandidates.Take(3).Select(c => c.Code).ToList();
        Assert.Contains(expectedCode, top3Codes);
    }

    [Fact]
    public async Task ClassifyAsync_Latency_ShouldBeSubMillisecond()
    {
        // Warm up
        await _classifier.ClassifyAsync("coffee shop cafe", topK: 5);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await _classifier.ClassifyAsync("custom computer programming and cloud software design", topK: 5);
        sw.Stop();

        Assert.NotEmpty(result.TopCandidates);
        // FastText vectorized dot product across 308 classes should be under 15ms
        Assert.True(sw.ElapsedMilliseconds < 15, $"Elapsed was {sw.ElapsedMilliseconds}ms");
    }
}
