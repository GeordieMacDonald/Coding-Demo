using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using NaicsClassifier.Api.Services;
using NaicsClassifier.Core.Models;
using Xunit;

namespace NaicsClassifier.Tests;

public class ApiIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ApiIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task GetModels_ShouldReturnTfIdfModel()
    {
        var response = await _client.GetAsync("/api/models");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var models = await response.Content.ReadFromJsonAsync<List<ClassifierMetadata>>();
        Assert.NotNull(models);
        Assert.True(models.Count >= 3);

        var tfidf = models.FirstOrDefault(m => m.Key == "tfidf");
        Assert.NotNull(tfidf);
        Assert.Contains("Microsoft.ML", tfidf.Name);

        var fasttext = models.FirstOrDefault(m => m.Key == "fasttext");
        Assert.NotNull(fasttext);

        var bert = models.FirstOrDefault(m => m.Key == "bert");
        Assert.NotNull(bert);
        Assert.Contains("BERT", bert.Name);
    }

    [Fact]
    public async Task Classify_WithBertAlgorithm_ShouldReturnExpectedResult()
    {
        var request = new ClassificationRequest
        {
            Text = "litigation advocacy and legal representation in court",
            Algorithm = "bert",
            TopK = 3
        };

        var response = await _client.PostAsJsonAsync("/api/classify", request);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var result = await response.Content.ReadFromJsonAsync<ClassificationResult>();
        Assert.NotNull(result);
        Assert.NotEmpty(result.TopCandidates);
        Assert.Contains("BERT", result.Algorithm);
        Assert.Equal("5411", result.TopCandidates[0].Code);
    }

    [Fact]
    public async Task Classify_ValidQuery_ShouldReturnCandidates()
    {
        var request = new ClassificationRequest
        {
            Text = "commercial roofing contractor and siding repair",
            TopK = 3
        };

        var response = await _client.PostAsJsonAsync("/api/classify", request);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var result = await response.Content.ReadFromJsonAsync<ClassificationResult>();
        Assert.NotNull(result);
        Assert.Equal(3, result.TopCandidates.Count);
        Assert.Equal("2381", result.TopCandidates[0].Code);
        Assert.NotEmpty(result.TopCandidates[0].MatchedTerms);
        Assert.True(result.ElapsedMilliseconds >= 0);
    }

    [Fact]
    public async Task Classify_EmptyQuery_ShouldReturnBadRequest()
    {
        var request = new ClassificationRequest
        {
            Text = "   ",
            TopK = 5
        };

        var response = await _client.PostAsJsonAsync("/api/classify", request);
        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
    }

    [Fact]
    public async Task GetNaicsByCode_ExistingCode_ShouldReturnDetails()
    {
        var response = await _client.GetAsync("/api/naics/5415");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var record = await response.Content.ReadFromJsonAsync<NaicsRecord>();
        Assert.NotNull(record);
        Assert.Equal("5415", record.Code);
        Assert.Contains("Computer Systems Design", record.Title);
        Assert.NotEmpty(record.IndexTerms);
    }

    [Fact]
    public async Task GetNaicsByCode_NonExistingCode_ShouldReturnNotFound()
    {
        var response = await _client.GetAsync("/api/naics/9999");
        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }

    [Fact]
    public async Task SearchNaics_ShouldReturnMatchingRecords()
    {
        var response = await _client.GetAsync("/api/naics/search?q=dentist&limit=5");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);

        var records = await response.Content.ReadFromJsonAsync<List<NaicsRecord>>();
        Assert.NotNull(records);
        Assert.NotEmpty(records);
        Assert.Contains(records, r => r.Code == "6212");
    }

    [Fact]
    public async Task GetStaticIndex_ShouldReturnHtmlDashboard()
    {
        var response = await _client.GetAsync("/index.html");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/html", response.Content.Headers.ContentType?.MediaType);

        var html = await response.Content.ReadAsStringAsync();
        Assert.Contains("NAICS Code Classification Engine", html);
    }
}
