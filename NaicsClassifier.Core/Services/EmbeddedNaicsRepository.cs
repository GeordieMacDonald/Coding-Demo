using System.Reflection;
using System.Text.Json;
using NaicsClassifier.Core.Models;

namespace NaicsClassifier.Core.Services;

public sealed class EmbeddedNaicsRepository : INaicsRepository
{
    private readonly Lazy<Task<List<NaicsRecord>>> _recordsTask;

    public EmbeddedNaicsRepository()
    {
        _recordsTask = new Lazy<Task<List<NaicsRecord>>>(LoadRecordsInternalAsync);
    }

    public async Task<IReadOnlyList<NaicsRecord>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        var records = await _recordsTask.Value;
        return records;
    }

    public async Task<NaicsRecord?> GetByCodeAsync(string code, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(code)) return null;
        var records = await GetAllAsync(cancellationToken);
        return records.FirstOrDefault(r => r.Code.Equals(code.Trim(), StringComparison.OrdinalIgnoreCase));
    }

    public async Task<IReadOnlyList<NaicsRecord>> SearchAsync(string term, int limit = 20, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(term))
        {
            var all = await GetAllAsync(cancellationToken);
            return all.Take(limit).ToList();
        }

        var normalized = term.Trim();
        var records = await GetAllAsync(cancellationToken);

        return records
            .Where(r => r.Code.StartsWith(normalized, StringComparison.OrdinalIgnoreCase) ||
                        r.Title.Contains(normalized, StringComparison.OrdinalIgnoreCase) ||
                        r.IndexTerms.Any(t => t.Contains(normalized, StringComparison.OrdinalIgnoreCase)))
            .Take(limit)
            .ToList();
    }

    private static async Task<List<NaicsRecord>> LoadRecordsInternalAsync()
    {
        var assembly = Assembly.GetExecutingAssembly();
        const string resourceName = "NaicsClassifier.Core.Data.naics_2022_4digit.json";

        using var stream = assembly.GetManifestResourceStream(resourceName)
            ?? throw new InvalidOperationException($"Embedded resource '{resourceName}' not found in assembly {assembly.FullName}.");

        var records = await JsonSerializer.DeserializeAsync<List<NaicsRecord>>(stream);
        return records ?? [];
    }
}
