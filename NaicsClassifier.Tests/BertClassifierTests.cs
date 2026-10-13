using NaicsClassifier.Core.Services;
using NaicsClassifier.Engine.Bert;
using Xunit;

namespace NaicsClassifier.Tests;

public class BertClassifierTests : IAsyncLifetime
{
    private readonly EmbeddedNaicsRepository _repository = new();
    private readonly BertClassifier _classifier = new();

    public async Task InitializeAsync()
    {
        var records = await _repository.GetAllAsync();
        await _classifier.InitializeAsync(records);
    }

    public Task DisposeAsync()
    {
        _classifier.Dispose();
        return Task.CompletedTask;
    }

    [Fact]
    public void Classifier_ShouldBeTrained()
    {
        Assert.True(_classifier.IsTrained);
        Assert.Equal("bert", _classifier.Key);
        Assert.Contains("BERT", _classifier.Name);
    }

    [Theory]
    [InlineData("commercial roofing contractor and siding installation", "2381")]
    [InlineData("poultry egg farming and chicken hatchery", "1123")]
    [InlineData("dentist office and teeth cleaning clinic", "6212")]
    [InlineData("pizza restaurant and food dining", "7225")]
    [InlineData("residential plumbing and furnace heating repair", "2382")]
    [InlineData("corporate attorney and contract legal advice", "5411")]
    public async Task ClassifyAsync_StandardQuery_ShouldIdentifyCorrectTopCode(string query, string expectedCode)
    {
        var result = await _classifier.ClassifyAsync(query, topK: 5);

        Assert.NotNull(result);
        Assert.NotEmpty(result.TopCandidates);
        Assert.Equal("BERT / Transformer (all-MiniLM-L6-v2)", result.Algorithm);

        var top = result.TopCandidates[0];
        Assert.Equal(expectedCode, top.Code);
        Assert.True(top.Confidence > 0);
        Assert.True(top.Score > 0.4);
    }

    [Theory]
    [InlineData("oral cavity hygiene and dental crowns", "6212")]
    [InlineData("litigation advocacy and legal representation in court", "5411")]
    [InlineData("water leak pipe repair and central air furnace maintenance", "2382")]
    public async Task ClassifyAsync_SemanticGeneralization_ShouldMatchConceptualSynonyms(string query, string expectedCode)
    {
        var result = await _classifier.ClassifyAsync(query, topK: 5);

        Assert.NotNull(result);
        Assert.NotEmpty(result.TopCandidates);

        var topCodes = result.TopCandidates.Take(3).Select(c => c.Code).ToList();
        Assert.Contains(expectedCode, topCodes);
    }

    [Fact]
    public async Task ClassifyAsync_EmptyQuery_ShouldReturnEmptyCandidates()
    {
        var result = await _classifier.ClassifyAsync("   ", topK: 5);

        Assert.NotNull(result);
        Assert.Empty(result.TopCandidates);
    }

    [Fact]
    public async Task ClassifyAsync_Latency_ShouldBeUnder50Ms()
    {
        // Warm up
        await _classifier.ClassifyAsync("coffee shop cafe", topK: 5);

        var sw = System.Diagnostics.Stopwatch.StartNew();
        var result = await _classifier.ClassifyAsync("custom computer programming and cloud software design", topK: 5);
        sw.Stop();

        Assert.NotEmpty(result.TopCandidates);
        // ONNX CPU inference + SIMD MaxSim search should easily run well under 50ms
        Assert.True(sw.ElapsedMilliseconds < 50, $"Elapsed was {sw.ElapsedMilliseconds}ms");
    }
}
