using NaicsClassifier.Core.Services;
using NaicsClassifier.Engine.Bert;
using NaicsClassifier.Engine.FastText;
using NaicsClassifier.Engine.TfIdf;
using Xunit;
using Xunit.Abstractions;

namespace NaicsClassifier.Tests;

public class BenchmarkServiceTests : IAsyncLifetime
{
    private readonly ITestOutputHelper _output;
    private readonly EmbeddedNaicsRepository _repository = new();
    private readonly TfIdfClassifier _tfidf = new();
    private readonly FastTextClassifier _fasttext = new();
    private readonly BertClassifier _bert = new();
    private readonly BenchmarkService _benchmarkService = new();

    public BenchmarkServiceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    public async Task InitializeAsync()
    {
        var records = await _repository.GetAllAsync();
        await _tfidf.InitializeAsync(records);
        await _fasttext.InitializeAsync(records);
        await _bert.InitializeAsync(records);
    }

    public Task DisposeAsync()
    {
        _bert.Dispose();
        return Task.CompletedTask;
    }

    [Fact]
    public void GetBenchmarkCases_ShouldReturnCuratedDataset()
    {
        var cases = _benchmarkService.GetBenchmarkCases();
        Assert.NotNull(cases);
        Assert.True(cases.Count >= 20);

        Assert.Contains(cases, c => c.Category == "Standard");
        Assert.Contains(cases, c => c.Category == "Semantic");
        Assert.Contains(cases, c => c.Category == "Typo");
    }

    [Fact]
    public async Task RunSuiteAsync_ShouldEvaluateClassifiersAndComputeAccuracies()
    {
        var classifiers = new INaicsClassifier[] { _tfidf, _fasttext, _bert };
        var result = await _benchmarkService.RunSuiteAsync(classifiers);

        Assert.NotNull(result);
        Assert.Equal(_benchmarkService.GetBenchmarkCases().Count, result.TotalCases);
        Assert.Equal(3, result.Models.Count);

        var tfidfSummary = result.Models.First(m => m.ModelKey == "tfidf");
        var fasttextSummary = result.Models.First(m => m.ModelKey == "fasttext");
        var bertSummary = result.Models.First(m => m.ModelKey == "bert");

        _output.WriteLine($"TF-IDF:   Top-1: {tfidfSummary.Top1Accuracy}%  Top-3: {tfidfSummary.Top3Accuracy}%  Latency: {tfidfSummary.AverageLatencyMs}ms  Typo: {tfidfSummary.CategoryAccuracies["Typo"]}%  Semantic: {tfidfSummary.CategoryAccuracies["Semantic"]}%");
        _output.WriteLine($"FastText: Top-1: {fasttextSummary.Top1Accuracy}%  Top-3: {fasttextSummary.Top3Accuracy}%  Latency: {fasttextSummary.AverageLatencyMs}ms  Typo: {fasttextSummary.CategoryAccuracies["Typo"]}%  Semantic: {fasttextSummary.CategoryAccuracies["Semantic"]}%");
        _output.WriteLine($"BERT:     Top-1: {bertSummary.Top1Accuracy}%  Top-3: {bertSummary.Top3Accuracy}%  Latency: {bertSummary.AverageLatencyMs}ms  Typo: {bertSummary.CategoryAccuracies["Typo"]}%  Semantic: {bertSummary.CategoryAccuracies["Semantic"]}%");

        // Verify key algorithmic traits:
        // 1. FastText handles typos better than TF-IDF
        Assert.True(fasttextSummary.CategoryAccuracies["Typo"] > tfidfSummary.CategoryAccuracies["Typo"]);

        // 2. BERT has superior semantic generalization
        Assert.True(bertSummary.CategoryAccuracies["Semantic"] >= tfidfSummary.CategoryAccuracies["Semantic"]);

        // 3. TF-IDF has lowest inference latency
        Assert.True(tfidfSummary.AverageLatencyMs < bertSummary.AverageLatencyMs);

        // 4. All models have positive Top-1 and high Top-3 accuracy
        Assert.True(tfidfSummary.Top1Accuracy >= 40);
        Assert.True(fasttextSummary.Top3Accuracy >= 60);
        Assert.True(bertSummary.Top1Accuracy >= 50);
    }
}
