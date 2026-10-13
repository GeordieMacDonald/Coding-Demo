using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NaicsClassifier.Core.Services;

namespace NaicsClassifier.Api.Services;

public sealed class ClassifierWarmupService : IHostedService
{
    private readonly INaicsRepository _repository;
    private readonly IEnumerable<INaicsClassifier> _classifiers;
    private readonly ILogger<ClassifierWarmupService> _logger;

    public ClassifierWarmupService(
        INaicsRepository repository,
        IEnumerable<INaicsClassifier> classifiers,
        ILogger<ClassifierWarmupService> logger)
    {
        _repository = repository;
        _classifiers = classifiers;
        _logger = logger;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("Loading NAICS dataset for classifier initialization...");
        var records = await _repository.GetAllAsync(cancellationToken);
        _logger.LogInformation("Loaded {Count} NAICS records.", records.Count);

        foreach (var classifier in _classifiers)
        {
            _logger.LogInformation("Initializing classifier '{Name}'...", classifier.Name);
            var sw = System.Diagnostics.Stopwatch.StartNew();
            await classifier.InitializeAsync(records, cancellationToken);
            sw.Stop();
            _logger.LogInformation("Classifier '{Name}' initialized in {ElapsedMs} ms.", classifier.Name, sw.ElapsedMilliseconds);
        }
    }

    public Task StopAsync(CancellationToken cancellationToken) => Task.CompletedTask;
}
