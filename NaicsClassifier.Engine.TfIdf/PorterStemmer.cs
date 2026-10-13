namespace NaicsClassifier.Engine.TfIdf;

/// <summary>
/// Standard Martin Porter English stemming algorithm implementation for normalising
/// English word inflections (plurals, tenses, suffixes) to common morphological roots.
/// </summary>
public static class PorterStemmer
{
    public static string StemText(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return string.Empty;

        var span = text.AsSpan();
        var sb = new System.Text.StringBuilder(text.Length);
        int i = 0;

        while (i < span.Length)
        {
            if (char.IsLetter(span[i]))
            {
                int start = i;
                while (i < span.Length && char.IsLetter(span[i]))
                {
                    i++;
                }
                var word = span.Slice(start, i - start).ToString().ToLowerInvariant();
                sb.Append(StemWord(word));
            }
            else
            {
                sb.Append(span[i]);
                i++;
            }
        }

        return sb.ToString();
    }

    public static string StemWord(string s)
    {
        if (s.Length <= 2) return s;

        char[] b = s.ToCharArray();
        int k = b.Length - 1;
        int k0 = 0;

        // Step 1a
        if (b[k] == 's')
        {
            if (EndsWith(b, k, "sses")) k -= 2;
            else if (EndsWith(b, k, "ies")) { b[k - 2] = 'i'; k -= 2; }
            else if (k >= 1 && b[k - 1] != 's') k--;
        }

        if (EndsWith(b, k, "eed"))
        {
            if (M(b, k0, k - 3) > 0) k--;
        }
        else if ((EndsWith(b, k, "ed") && VowelInStem(b, k0, k - 2)) ||
                 (EndsWith(b, k, "ing") && VowelInStem(b, k0, k - 3)))
        {
            k = EndsWith(b, k, "ed") ? k - 2 : k - 3;
            if (EndsWith(b, k, "at")) { b[k++] = 'a'; b[k++] = 't'; b[k] = 'e'; }
            else if (EndsWith(b, k, "bl")) { b[k++] = 'b'; b[k++] = 'l'; b[k] = 'e'; }
            else if (EndsWith(b, k, "iz")) { b[k++] = 'i'; b[k++] = 'z'; b[k] = 'e'; }
            else if (DoubleConsonant(b, k))
            {
                char ch = b[k--];
                if (ch == 'l' || ch == 's' || ch == 'z') k++;
            }
            else if (M(b, k0, k) == 1 && Cvc(b, k))
            {
                b[++k] = 'e';
            }
        }

        // Step 1c
        if (b[k] == 'y' && VowelInStem(b, k0, k - 1))
        {
            b[k] = 'i';
        }

        // Step 2
        switch (b[k - 1])
        {
            case 'a':
                if (EndsWith(b, k, "ational") && M(b, k0, k - 7) > 0) { Setto(b, ref k, "ate"); break; }
                if (EndsWith(b, k, "tional") && M(b, k0, k - 6) > 0) { Setto(b, ref k, "tion"); break; }
                break;
            case 'c':
                if (EndsWith(b, k, "enci") && M(b, k0, k - 4) > 0) { Setto(b, ref k, "ence"); break; }
                if (EndsWith(b, k, "anci") && M(b, k0, k - 4) > 0) { Setto(b, ref k, "ance"); break; }
                break;
            case 'e':
                if (EndsWith(b, k, "izer") && M(b, k0, k - 4) > 0) { Setto(b, ref k, "ize"); break; }
                break;
            case 'l':
                if (EndsWith(b, k, "bli") && M(b, k0, k - 3) > 0) { Setto(b, ref k, "ble"); break; }
                if (EndsWith(b, k, "alli") && M(b, k0, k - 4) > 0) { Setto(b, ref k, "al"); break; }
                if (EndsWith(b, k, "entli") && M(b, k0, k - 5) > 0) { Setto(b, ref k, "ent"); break; }
                if (EndsWith(b, k, "eli") && M(b, k0, k - 3) > 0) { Setto(b, ref k, "e"); break; }
                if (EndsWith(b, k, "ousli") && M(b, k0, k - 5) > 0) { Setto(b, ref k, "ous"); break; }
                break;
            case 'o':
                if (EndsWith(b, k, "ization") && M(b, k0, k - 7) > 0) { Setto(b, ref k, "ize"); break; }
                if (EndsWith(b, k, "ation") && M(b, k0, k - 5) > 0) { Setto(b, ref k, "ate"); break; }
                if (EndsWith(b, k, "ator") && M(b, k0, k - 4) > 0) { Setto(b, ref k, "ate"); break; }
                break;
            case 's':
                if (EndsWith(b, k, "alism") && M(b, k0, k - 5) > 0) { Setto(b, ref k, "al"); break; }
                if (EndsWith(b, k, "iveness") && M(b, k0, k - 7) > 0) { Setto(b, ref k, "ive"); break; }
                if (EndsWith(b, k, "fulness") && M(b, k0, k - 7) > 0) { Setto(b, ref k, "ful"); break; }
                if (EndsWith(b, k, "ousness") && M(b, k0, k - 7) > 0) { Setto(b, ref k, "ous"); break; }
                break;
            case 't':
                if (EndsWith(b, k, "aliti") && M(b, k0, k - 5) > 0) { Setto(b, ref k, "al"); break; }
                if (EndsWith(b, k, "iviti") && M(b, k0, k - 5) > 0) { Setto(b, ref k, "ive"); break; }
                break;
        }

        // Step 4
        if (EndsWith(b, k, "al") && M(b, k0, k - 2) > 1) k -= 2;
        else if (EndsWith(b, k, "ance") && M(b, k0, k - 4) > 1) k -= 4;
        else if (EndsWith(b, k, "ence") && M(b, k0, k - 4) > 1) k -= 4;
        else if (EndsWith(b, k, "er") && M(b, k0, k - 2) > 1) k -= 2;
        else if (EndsWith(b, k, "ic") && M(b, k0, k - 2) > 1) k -= 2;
        else if (EndsWith(b, k, "able") && M(b, k0, k - 4) > 1) k -= 4;
        else if (EndsWith(b, k, "ible") && M(b, k0, k - 4) > 1) k -= 4;
        else if (EndsWith(b, k, "ant") && M(b, k0, k - 3) > 1) k -= 3;
        else if (EndsWith(b, k, "ement") && M(b, k0, k - 5) > 1) k -= 5;
        else if (EndsWith(b, k, "ment") && M(b, k0, k - 4) > 1) k -= 4;
        else if (EndsWith(b, k, "ent") && M(b, k0, k - 3) > 1) k -= 3;
        else if ((EndsWith(b, k, "sion") || EndsWith(b, k, "tion")) && M(b, k0, k - 3) > 1) k -= 3;
        else if (EndsWith(b, k, "ou") && M(b, k0, k - 2) > 1) k -= 2;
        else if (EndsWith(b, k, "ism") && M(b, k0, k - 3) > 1) k -= 3;
        else if (EndsWith(b, k, "ate") && M(b, k0, k - 3) > 1) k -= 3;
        else if (EndsWith(b, k, "iti") && M(b, k0, k - 3) > 1) k -= 3;
        else if (EndsWith(b, k, "ous") && M(b, k0, k - 3) > 1) k -= 3;
        else if (EndsWith(b, k, "ive") && M(b, k0, k - 3) > 1) k -= 3;
        else if (EndsWith(b, k, "ize") && M(b, k0, k - 3) > 1) k -= 3;

        // Step 5a
        if (b[k] == 'e')
        {
            int a = M(b, k0, k - 1);
            if (a > 1 || (a == 1 && !Cvc(b, k - 1))) k--;
        }

        // Step 5b
        if (b[k] == 'l' && DoubleConsonant(b, k) && M(b, k0, k - 1) > 1) k--;

        return new string(b, 0, k + 1);
    }

    private static bool Consonant(char[] b, int i)
    {
        return b[i] switch
        {
            'a' or 'e' or 'i' or 'o' or 'u' => false,
            'y' => i == 0 || !Consonant(b, i - 1),
            _ => true
        };
    }

    private static int M(char[] b, int k0, int k)
    {
        int n = 0;
        int i = k0;
        while (true)
        {
            if (i > k) return n;
            if (!Consonant(b, i)) break;
            i++;
        }
        i++;
        while (true)
        {
            while (true)
            {
                if (i > k) return n;
                if (Consonant(b, i)) break;
                i++;
            }
            i++;
            n++;
            while (true)
            {
                if (i > k) return n;
                if (!Consonant(b, i)) break;
                i++;
            }
            i++;
        }
    }

    private static bool VowelInStem(char[] b, int k0, int k)
    {
        for (int i = k0; i <= k; i++)
        {
            if (!Consonant(b, i)) return true;
        }
        return false;
    }

    private static bool DoubleConsonant(char[] b, int i)
    {
        if (i < 1) return false;
        if (b[i] != b[i - 1]) return false;
        return Consonant(b, i);
    }

    private static bool Cvc(char[] b, int i)
    {
        if (i < 2 || !Consonant(b, i) || Consonant(b, i - 1) || !Consonant(b, i - 2)) return false;
        char ch = b[i];
        return ch != 'w' && ch != 'x' && ch != 'y';
    }

    private static bool EndsWith(char[] b, int k, string s)
    {
        int l = s.Length;
        int o = k - l + 1;
        if (o < 0) return false;
        for (int i = 0; i < l; i++)
        {
            if (b[o + i] != s[i]) return false;
        }
        return true;
    }

    private static void Setto(char[] b, ref int k, string s)
    {
        int l = s.Length;
        int o = k - s.Length + 1;
        for (int i = 0; i < l; i++) b[o + i] = s[i];
        k = o + l - 1;
    }
}
