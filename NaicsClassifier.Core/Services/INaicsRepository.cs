using NaicsClassifier.Core.Models;

namespace NaicsClassifier.Core.Services;

public interface INaicsRepository
{
    Task<IReadOnlyList<NaicsRecord>> GetAllAsync(CancellationToken cancellationToken = default);
    Task<NaicsRecord?> GetByCodeAsync(string code, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<NaicsRecord>> SearchAsync(string term, int limit = 20, CancellationToken cancellationToken = default);
}
