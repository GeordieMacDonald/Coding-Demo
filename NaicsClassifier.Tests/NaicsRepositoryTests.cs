using NaicsClassifier.Core.Services;
using Xunit;

namespace NaicsClassifier.Tests;

public class NaicsRepositoryTests
{
    [Fact]
    public async Task GetAllAsync_ShouldLoadAll308IndustryGroups()
    {
        var repo = new EmbeddedNaicsRepository();
        var records = await repo.GetAllAsync();

        Assert.NotNull(records);
        Assert.Equal(308, records.Count);

        // Verify each record has 4 digits and non-empty title/sector
        foreach (var record in records)
        {
            Assert.Equal(4, record.Code.Length);
            Assert.False(string.IsNullOrWhiteSpace(record.Title));
            Assert.False(string.IsNullOrWhiteSpace(record.SectorCode));
            Assert.False(string.IsNullOrWhiteSpace(record.SectorTitle));
            Assert.False(string.IsNullOrWhiteSpace(record.AggregatedContent));
        }
    }

    [Theory]
    [InlineData("5415", "Computer Systems Design and Related Services", "54")]
    [InlineData("2381", "Foundation, Structure, and Building Exterior Contractors", "23")]
    [InlineData("7225", "Restaurants and Other Eating Places", "72")]
    public async Task GetByCodeAsync_ShouldReturnExpectedRecord(string code, string expectedTitlePart, string expectedSector)
    {
        var repo = new EmbeddedNaicsRepository();
        var record = await repo.GetByCodeAsync(code);

        Assert.NotNull(record);
        Assert.Equal(code, record.Code);
        Assert.Contains(expectedTitlePart, record.Title);
        Assert.Equal(expectedSector, record.SectorCode);
        Assert.NotEmpty(record.IndexTerms);
    }
}
