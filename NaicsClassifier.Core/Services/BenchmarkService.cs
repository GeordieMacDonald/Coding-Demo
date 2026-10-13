using System.Diagnostics;
using NaicsClassifier.Core.Models;

namespace NaicsClassifier.Core.Services;

public interface IBenchmarkService
{
    IReadOnlyList<BenchmarkCase> GetBenchmarkCases();
    Task<BenchmarkSuiteResult> RunSuiteAsync(IEnumerable<INaicsClassifier> classifiers, CancellationToken ct = default);
}

public class BenchmarkService : IBenchmarkService
{
    private static readonly List<BenchmarkCase> Cases = new()
    {
        // Standard Queries
        new BenchmarkCase
        {
            Query = "commercial roofing contractor and siding repair",
            ExpectedCode = "2381",
            ExpectedTitle = "Foundation, Structure, and Building Exterior Contractors",
            Category = "Standard",
            Note = "Direct building exterior trade activity"
        },
        new BenchmarkCase
        {
            Query = "poultry hatchery and chicken egg farming",
            ExpectedCode = "1123",
            ExpectedTitle = "Poultry and Egg Production",
            Category = "Standard",
            Note = "Agricultural livestock domain"
        },
        new BenchmarkCase
        {
            Query = "dentist office and dental hygiene clinic",
            ExpectedCode = "6212",
            ExpectedTitle = "Offices of Dentists",
            Category = "Standard",
            Note = "Direct dental practice description"
        },
        new BenchmarkCase
        {
            Query = "full-service italian restaurant and pizzeria dining",
            ExpectedCode = "7225",
            ExpectedTitle = "Restaurants and Other Eating Places",
            Category = "Standard",
            Note = "Food service and dining establishments"
        },
        new BenchmarkCase
        {
            Query = "residential plumbing and furnace heating repair",
            ExpectedCode = "2382",
            ExpectedTitle = "Building Equipment Contractors",
            Category = "Standard",
            Note = "HVAC and mechanical trade"
        },
        new BenchmarkCase
        {
            Query = "custom computer software programming and web application design",
            ExpectedCode = "5415",
            ExpectedTitle = "Computer Systems Design and Related Services",
            Category = "Standard",
            Note = "IT and custom programming services"
        },
        new BenchmarkCase
        {
            Query = "corporate legal counseling and attorney services",
            ExpectedCode = "5411",
            ExpectedTitle = "Legal Services",
            Category = "Standard",
            Note = "Law practice and legal advice"
        },
        new BenchmarkCase
        {
            Query = "general medical and surgical hospital services",
            ExpectedCode = "6221",
            ExpectedTitle = "General Medical and Surgical Hospitals",
            Category = "Standard",
            Note = "Inpatient acute healthcare facility"
        },
        new BenchmarkCase
        {
            Query = "new car dealership and automobile sales",
            ExpectedCode = "4411",
            ExpectedTitle = "Automobile Dealers",
            Category = "Standard",
            Note = "Retail motor vehicle dealership"
        },
        new BenchmarkCase
        {
            Query = "commercial dry cleaning and laundry services",
            ExpectedCode = "8123",
            ExpectedTitle = "Drycleaning and Laundry Services",
            Category = "Standard",
            Note = "Personal garment care services"
        },

        // Semantic / Conceptual Generalization (Synonyms & Nuance)
        new BenchmarkCase
        {
            Query = "litigation advocacy and legal representation in court",
            ExpectedCode = "5411",
            ExpectedTitle = "Legal Services",
            Category = "Semantic",
            Note = "Courtroom advocacy without exact 'lawyer/attorney' token"
        },
        new BenchmarkCase
        {
            Query = "oral cavity hygiene and dental crowns",
            ExpectedCode = "6212",
            ExpectedTitle = "Offices of Dentists",
            Category = "Semantic",
            Note = "Clinical dentistry procedures without the word 'office'"
        },
        new BenchmarkCase
        {
            Query = "water leak pipe repair and central air furnace maintenance",
            ExpectedCode = "2382",
            ExpectedTitle = "Building Equipment Contractors",
            Category = "Semantic",
            Note = "Plumbing/HVAC trade described through operational symptoms"
        },
        new BenchmarkCase
        {
            Query = "canine vaccination and companion animal wellness examinations",
            ExpectedCode = "5419",
            ExpectedTitle = "Other Professional, Scientific, and Technical Services",
            Category = "Semantic",
            Note = "Veterinary clinic procedures (under 5419)"
        },
        new BenchmarkCase
        {
            Query = "fine dining steakhouse and culinary wine bar",
            ExpectedCode = "7225",
            ExpectedTitle = "Restaurants and Other Eating Places",
            Category = "Semantic",
            Note = "High-end dining synonyms"
        },

        // Typo & Misspelling Resilience
        new BenchmarkCase
        {
            Query = "comercial roofin contrctor and sidng repare",
            ExpectedCode = "2381",
            ExpectedTitle = "Foundation, Structure, and Building Exterior Contractors",
            Category = "Typo",
            Note = "Multiple simultaneous misspellings"
        },
        new BenchmarkCase
        {
            Query = "sofware prgramming and saas developmnt",
            ExpectedCode = "5415",
            ExpectedTitle = "Computer Systems Design and Related Services",
            Category = "Typo",
            Note = "Spelling errors in tech terminology"
        },
        new BenchmarkCase
        {
            Query = "plumbin and heatin hvac contrctor",
            ExpectedCode = "2382",
            ExpectedTitle = "Building Equipment Contractors",
            Category = "Typo",
            Note = "Colloquial dropped-g endings and typos"
        },
        new BenchmarkCase
        {
            Query = "famly resturant and pizza dinng",
            ExpectedCode = "7225",
            ExpectedTitle = "Restaurants and Other Eating Places",
            Category = "Typo",
            Note = "Common casual misspellings"
        },
        new BenchmarkCase
        {
            Query = "poultri hatcheri and egg farmng",
            ExpectedCode = "1123",
            ExpectedTitle = "Poultry and Egg Production",
            Category = "Typo",
            Note = "Phonetic substitution errors"
        },
        new BenchmarkCase
        {
            Query = "dentst ofice and teath clinik",
            ExpectedCode = "6212",
            ExpectedTitle = "Offices of Dentists",
            Category = "Typo",
            Note = "Heavy phonetic typo query"
        },

        // Ambiguous / Multi-Activity
        new BenchmarkCase
        {
            Query = "security alarm installation and residential monitoring",
            ExpectedCode = "5616",
            ExpectedTitle = "Investigation and Security Services",
            Category = "Ambiguous",
            Note = "Installation trade with ongoing monitoring service"
        }
    };

    public IReadOnlyList<BenchmarkCase> GetBenchmarkCases() => Cases;

    public async Task<BenchmarkSuiteResult> RunSuiteAsync(IEnumerable<INaicsClassifier> classifiers, CancellationToken ct = default)
    {
        var classifierList = classifiers.ToList();
        var evaluations = new List<BenchmarkCaseEvaluation>();
        var categories = Cases.Select(c => c.Category).Distinct().ToList();

        // Track model stats
        var modelStats = classifierList.ToDictionary(
            c => c.Key,
            c => new
            {
                Classifier = c,
                Latencies = new List<double>(),
                Top1Hits = 0,
                Top3Hits = 0,
                Top5Hits = 0,
                CatTop1Hits = categories.ToDictionary(cat => cat, _ => 0),
                CatTotals = categories.ToDictionary(cat => cat, _ => 0)
            });

        // Run cases
        foreach (var benchmarkCase in Cases)
        {
            ct.ThrowIfCancellationRequested();
            var caseEval = new BenchmarkCaseEvaluation
            {
                CaseInfo = benchmarkCase
            };

            foreach (var classifier in classifierList)
            {
                var sw = Stopwatch.StartNew();
                var result = await classifier.ClassifyAsync(benchmarkCase.Query, topK: 5, ct);
                sw.Stop();

                double elapsedMs = Math.Round(sw.Elapsed.TotalMilliseconds, 2);
                var candidates = result.TopCandidates;

                bool isTop1 = candidates.Count > 0 && candidates[0].Code == benchmarkCase.ExpectedCode;
                bool isTop3 = candidates.Take(3).Any(c => c.Code == benchmarkCase.ExpectedCode);
                bool isTop5 = candidates.Take(5).Any(c => c.Code == benchmarkCase.ExpectedCode);

                var topCand = candidates.Count > 0 ? candidates[0] : null;

                caseEval.Results[classifier.Key] = new CaseModelResult
                {
                    ModelKey = classifier.Key,
                    PredictedCode = topCand?.Code ?? string.Empty,
                    PredictedTitle = topCand?.Title ?? string.Empty,
                    Confidence = topCand?.Confidence ?? 0,
                    Score = topCand?.Score ?? 0,
                    IsTop1 = isTop1,
                    IsTop3 = isTop3,
                    IsTop5 = isTop5,
                    ElapsedMs = elapsedMs
                };

                var stat = modelStats[classifier.Key];
                stat.Latencies.Add(elapsedMs);
                stat.CatTotals[benchmarkCase.Category]++;

                if (isTop1)
                {
                    // Use reflection or mutable struct/class
                }
            }

            evaluations.Add(caseEval);
        }

        // Aggregate summaries
        var summaries = new List<ModelBenchmarkSummary>();
        foreach (var classifier in classifierList)
        {
            int top1 = evaluations.Count(e => e.Results.TryGetValue(classifier.Key, out var r) && r.IsTop1);
            int top3 = evaluations.Count(e => e.Results.TryGetValue(classifier.Key, out var r) && r.IsTop3);
            int top5 = evaluations.Count(e => e.Results.TryGetValue(classifier.Key, out var r) && r.IsTop5);

            var latencies = evaluations
                .Where(e => e.Results.ContainsKey(classifier.Key))
                .Select(e => e.Results[classifier.Key].ElapsedMs)
                .ToList();

            var catAccuracies = new Dictionary<string, double>();
            foreach (var cat in categories)
            {
                int catTotal = Cases.Count(c => c.Category == cat);
                int catTop1 = evaluations.Count(e => e.CaseInfo.Category == cat && e.Results.TryGetValue(classifier.Key, out var r) && r.IsTop1);
                catAccuracies[cat] = catTotal > 0 ? Math.Round((double)catTop1 / catTotal * 100.0, 1) : 0;
            }

            summaries.Add(new ModelBenchmarkSummary
            {
                ModelKey = classifier.Key,
                ModelName = classifier.Name,
                TotalQueries = Cases.Count,
                Top1Hits = top1,
                Top1Accuracy = Math.Round((double)top1 / Cases.Count * 100.0, 1),
                Top3Hits = top3,
                Top3Accuracy = Math.Round((double)top3 / Cases.Count * 100.0, 1),
                Top5Hits = top5,
                Top5Accuracy = Math.Round((double)top5 / Cases.Count * 100.0, 1),
                AverageLatencyMs = latencies.Count > 0 ? Math.Round(latencies.Average(), 2) : 0,
                MinLatencyMs = latencies.Count > 0 ? Math.Round(latencies.Min(), 2) : 0,
                MaxLatencyMs = latencies.Count > 0 ? Math.Round(latencies.Max(), 2) : 0,
                CategoryAccuracies = catAccuracies
            });
        }

        return new BenchmarkSuiteResult
        {
            ExecutedAt = DateTime.UtcNow,
            TotalCases = Cases.Count,
            Categories = categories,
            Models = summaries,
            Evaluations = evaluations
        };
    }
}
