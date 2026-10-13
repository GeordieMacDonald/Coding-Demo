using NaicsClassifier.Core.Models;

namespace NaicsClassifier.Core.Services;

public interface INaicsClassifier
{
    string Key { get; }
    string Name { get; }
    string Description { get; }
    bool IsTrained { get; }

    Task InitializeAsync(IReadOnlyList<NaicsRecord> records, CancellationToken cancellationToken = default);
    Task<ClassificationResult> ClassifyAsync(string query, int topK = 5, CancellationToken cancellationToken = default);
}
