using Microsoft.AspNetCore.Mvc;
using NaicsClassifier.Api.Services;
using NaicsClassifier.Core.Models;
using NaicsClassifier.Core.Services;
using NaicsClassifier.Engine.Bert;
using NaicsClassifier.Engine.FastText;
using NaicsClassifier.Engine.TfIdf;

var builder = WebApplication.CreateBuilder(args);

// Register Core & Engine services
builder.Services.AddSingleton<INaicsRepository, EmbeddedNaicsRepository>();
builder.Services.AddSingleton<INaicsClassifier, TfIdfClassifier>();
builder.Services.AddSingleton<INaicsClassifier, FastTextClassifier>();
builder.Services.AddSingleton<INaicsClassifier, BertClassifier>();
builder.Services.AddSingleton<IClassifierRegistry, ClassifierRegistry>();
builder.Services.AddSingleton<IBenchmarkService, BenchmarkService>();
builder.Services.AddHostedService<ClassifierWarmupService>();

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

var app = builder.Build();

app.UseCors();
app.UseDefaultFiles();
app.UseStaticFiles();

// API: List Available Classification Models
app.MapGet("/api/models", (IClassifierRegistry registry) =>
{
    return Results.Ok(registry.List());
});

// API: Classify Business Activity Query
app.MapPost("/api/classify", async (
    [FromBody] ClassificationRequest request,
    [FromServices] IClassifierRegistry registry,
    CancellationToken ct) =>
{
    if (string.IsNullOrWhiteSpace(request.Text))
    {
        return Results.BadRequest(new { error = "Text parameter is required." });
    }

    var classifier = registry.Get(request.Algorithm);
    if (classifier == null)
    {
        return Results.NotFound(new { error = $"Classifier '{request.Algorithm}' not found." });
    }

    if (!classifier.IsTrained)
    {
        return Results.StatusCode(StatusCodes.Status503ServiceUnavailable);
    }

    int topK = Math.Clamp(request.TopK, 1, 20);
    var result = await classifier.ClassifyAsync(request.Text, topK, ct);

    if (request.MinScore > 0)
    {
        result = new ClassificationResult
        {
            Query = result.Query,
            Algorithm = result.Algorithm,
            ElapsedMilliseconds = result.ElapsedMilliseconds,
            TopCandidates = result.TopCandidates.Where(c => c.Score >= request.MinScore).ToList()
        };
    }

    return Results.Ok(result);
});

// API: Get NAICS Details by 4-digit Code
app.MapGet("/api/naics/{code}", async (
    string code,
    [FromServices] INaicsRepository repository,
    CancellationToken ct) =>
{
    var record = await repository.GetByCodeAsync(code, ct);
    if (record == null)
    {
        return Results.NotFound(new { error = $"NAICS code '{code}' not found." });
    }

    return Results.Ok(record);
});

// API: Search NAICS records by keyword
app.MapGet("/api/naics/search", async (
    [FromQuery] string? q,
    [FromQuery] int limit,
    [FromServices] INaicsRepository repository,
    CancellationToken ct) =>
{
    int safeLimit = limit <= 0 ? 20 : Math.Min(limit, 50);
    var results = await repository.SearchAsync(q ?? string.Empty, safeLimit, ct);
    return Results.Ok(results);
});

// API: Get Benchmark Cases
app.MapGet("/api/benchmark/cases", ([FromServices] IBenchmarkService benchmarkService) =>
{
    return Results.Ok(benchmarkService.GetBenchmarkCases());
});

// API: Run Benchmark Suite Across All Registered Models
app.MapPost("/api/benchmark/run", async (
    [FromServices] IBenchmarkService benchmarkService,
    [FromServices] IClassifierRegistry registry,
    CancellationToken ct) =>
{
    var classifiers = registry.GetAll().Where(c => c.IsTrained);
    var suiteResult = await benchmarkService.RunSuiteAsync(classifiers, ct);
    return Results.Ok(suiteResult);
});

app.Run();

public partial class Program { }
