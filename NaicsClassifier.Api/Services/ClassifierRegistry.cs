using NaicsClassifier.Core.Services;

namespace NaicsClassifier.Api.Services;

public interface IClassifierRegistry
{
    void Register(INaicsClassifier classifier);
    INaicsClassifier? Get(string? key);
    IReadOnlyList<INaicsClassifier> GetAll();
    IReadOnlyList<ClassifierMetadata> List();
}

public sealed record ClassifierMetadata(
    string Key,
    string Name,
    string Description,
    bool IsReady);

public sealed class ClassifierRegistry : IClassifierRegistry
{
    private readonly Dictionary<string, INaicsClassifier> _classifiers = new(StringComparer.OrdinalIgnoreCase);

    public ClassifierRegistry(IEnumerable<INaicsClassifier> classifiers)
    {
        foreach (var c in classifiers)
        {
            Register(c);
        }
    }

    public void Register(INaicsClassifier classifier)
    {
        _classifiers[classifier.Key] = classifier;
    }

    public INaicsClassifier? Get(string? key)
    {
        if (string.IsNullOrWhiteSpace(key))
        {
            return _classifiers.Values.FirstOrDefault();
        }

        return _classifiers.GetValueOrDefault(key) ?? _classifiers.Values.FirstOrDefault();
    }

    public IReadOnlyList<INaicsClassifier> GetAll()
    {
        return _classifiers.Values.ToList();
    }

    public IReadOnlyList<ClassifierMetadata> List()
    {
        return _classifiers.Values
            .Select(c => new ClassifierMetadata(c.Key, c.Name, c.Description, c.IsTrained))
            .ToList();
    }
}
