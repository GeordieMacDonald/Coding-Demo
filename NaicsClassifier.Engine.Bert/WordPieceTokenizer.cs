using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace NaicsClassifier.Engine.Bert;

/// <summary>
/// Fast WordPiece tokenizer compatible with BERT / all-MiniLM-L6-v2 vocabulary.
/// </summary>
public class WordPieceTokenizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.OrdinalIgnoreCase);
    private int _unkId = 100;
    private int _clsId = 101;
    private int _sepId = 102;
    private int _padId = 0;

    public int UnkId => _unkId;
    public int ClsId => _clsId;
    public int SepId => _sepId;
    public int PadId => _padId;
    public int VocabSize => _vocab.Count;

    public WordPieceTokenizer(string vocabPath)
    {
        using var stream = File.OpenRead(vocabPath);
        LoadFromStream(stream);
    }

    public WordPieceTokenizer(Stream vocabStream)
    {
        LoadFromStream(vocabStream);
    }

    private void LoadFromStream(Stream stream)
    {
        using var reader = new StreamReader(stream, Encoding.UTF8);
        string? line;
        int id = 0;
        while ((line = reader.ReadLine()) != null)
        {
            var token = line.Trim();
            if (!string.IsNullOrEmpty(token))
            {
                _vocab[token] = id;
            }
            id++;
        }

        _vocab.TryGetValue("[UNK]", out _unkId);
        _vocab.TryGetValue("[CLS]", out _clsId);
        _vocab.TryGetValue("[SEP]", out _sepId);
        _vocab.TryGetValue("[PAD]", out _padId);
    }

    /// <summary>
    /// Tokenizes input text into token IDs including [CLS] and [SEP].
    /// </summary>
    public long[] Tokenize(string text, int maxTokens = 128)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return new long[] { _clsId, _sepId };
        }

        var tokens = new List<long> { _clsId };
        var words = BasicTokenize(text);

        foreach (var word in words)
        {
            if (tokens.Count >= maxTokens - 1) break;

            if (_vocab.TryGetValue(word, out int exactId))
            {
                tokens.Add(exactId);
                continue;
            }

            int start = 0;
            bool isBad = false;
            var subTokens = new List<int>();

            while (start < word.Length)
            {
                int end = word.Length;
                int curSubToken = -1;

                while (start < end)
                {
                    string sub = (start == 0) ? word.Substring(start, end - start) : "##" + word.Substring(start, end - start);
                    if (_vocab.TryGetValue(sub, out int subId))
                    {
                        curSubToken = subId;
                        break;
                    }
                    end--;
                }

                if (curSubToken == -1)
                {
                    isBad = true;
                    break;
                }

                subTokens.Add(curSubToken);
                start = end;
            }

            if (isBad)
            {
                tokens.Add(_unkId);
            }
            else
            {
                foreach (var st in subTokens)
                {
                    if (tokens.Count < maxTokens - 1)
                        tokens.Add(st);
                }
            }
        }

        tokens.Add(_sepId);
        return tokens.ToArray();
    }

    public List<string> TokenizeToWords(string text)
    {
        return BasicTokenize(text);
    }

    private static List<string> BasicTokenize(string text)
    {
        var words = new List<string>();
        var cur = new StringBuilder();

        foreach (char c in text.ToLowerInvariant())
        {
            if (char.IsLetterOrDigit(c))
            {
                cur.Append(c);
            }
            else
            {
                if (cur.Length > 0)
                {
                    words.Add(cur.ToString());
                    cur.Clear();
                }
                if (!char.IsWhiteSpace(c))
                {
                    words.Add(c.ToString());
                }
            }
        }
        if (cur.Length > 0)
        {
            words.Add(cur.ToString());
        }

        return words;
    }
}
