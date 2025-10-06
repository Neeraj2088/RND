using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas.Parser;
using iText.Kernel.Pdf.Canvas.Parser.Listener;
using System.Text.Json;
using System.Text.RegularExpressions;

// Install NuGet package: itext7

public class PdfSection
{
    public string Type { get; set; } // "heading" or "content"
    public int Level { get; set; } // 1 for main heading, 2 for subheading, etc.
    public string Text { get; set; }
    public List<PdfSection> Children { get; set; } = new List<PdfSection>();
}

public class PdfParserService
{
    public string ParsePdfToJson(string filePath)
    {
        var rootSections = ParsePdf(filePath);
        var options = new JsonSerializerOptions { WriteIndented = true };
        return JsonSerializer.Serialize(rootSections, options);
    }

    public List<PdfSection> ParsePdf(string filePath)
    {
        var allLines = ExtractTextLines(filePath);
        return BuildHierarchy(allLines);
    }

    private List<string> ExtractTextLines(string filePath)
    {
        var lines = new List<string>();
        
        using (PdfReader reader = new PdfReader(filePath))
        using (PdfDocument pdfDoc = new PdfDocument(reader))
        {
            for (int i = 1; i <= pdfDoc.GetNumberOfPages(); i++)
            {
                var page = pdfDoc.GetPage(i);
                var strategy = new SimpleTextExtractionStrategy();
                string pageText = PdfTextExtractor.GetTextFromPage(page, strategy);
                
                var pageLines = pageText.Split('\n')
                    .Select(l => l.Trim())
                    .Where(l => !string.IsNullOrWhiteSpace(l))
                    .ToList();
                
                lines.AddRange(pageLines);
            }
        }
        
        return lines;
    }

    private List<PdfSection> BuildHierarchy(List<string> lines)
    {
        var root = new List<PdfSection>();
        var stack = new Stack<PdfSection>();
        var contentBuffer = new List<string>();
        
        foreach (var line in lines)
        {
            int headingLevel = DetectHeadingLevel(line);
            
            if (headingLevel > 0)
            {
                // Add buffered content to current section
                if (contentBuffer.Count > 0 && stack.Count > 0)
                {
                    var content = new PdfSection
                    {
                        Type = "content",
                        Level = stack.Peek().Level,
                        Text = string.Join(" ", contentBuffer)
                    };
                    stack.Peek().Children.Add(content);
                    contentBuffer.Clear();
                }
                
                var heading = new PdfSection
                {
                    Type = "heading",
                    Level = headingLevel,
                    Text = CleanHeadingText(line)
                };
                
                // Pop stack until we find parent level
                while (stack.Count > 0 && stack.Peek().Level >= headingLevel)
                {
                    stack.Pop();
                }
                
                // Add to appropriate parent
                if (stack.Count == 0)
                {
                    root.Add(heading);
                }
                else
                {
                    stack.Peek().Children.Add(heading);
                }
                
                stack.Push(heading);
            }
            else
            {
                // Regular content
                contentBuffer.Add(line);
            }
        }
        
        // Add any remaining content
        if (contentBuffer.Count > 0 && stack.Count > 0)
        {
            var content = new PdfSection
            {
                Type = "content",
                Level = stack.Peek().Level,
                Text = string.Join(" ", contentBuffer)
            };
            stack.Peek().Children.Add(content);
        }
        
        return root;
    }

    private int DetectHeadingLevel(string line)
    {
        // Pattern 1: Numbered headings (1., 1.1., 1.1.1., etc.)
        var numberedMatch = Regex.Match(line, @"^(\d+\.)+\s+");
        if (numberedMatch.Success)
        {
            return numberedMatch.Value.Count(c => c == '.');
        }
        
        // Pattern 2: Chapter/Section keywords
        if (Regex.IsMatch(line, @"^(Chapter|CHAPTER|Section|SECTION)\s+\d+", RegexOptions.IgnoreCase))
        {
            return 1;
        }
        
        // Pattern 3: All caps (likely heading)
        if (line.Length < 100 && line == line.ToUpper() && line.Any(char.IsLetter))
        {
            return 2;
        }
        
        // Pattern 4: Title case and short (likely heading)
        if (line.Length < 80 && char.IsUpper(line[0]) && !line.EndsWith(".") && !line.Contains("  "))
        {
            var words = line.Split(' ');
            int upperCaseCount = words.Count(w => w.Length > 0 && char.IsUpper(w[0]));
            if (upperCaseCount >= words.Length * 0.7)
            {
                return 3;
            }
        }
        
        return 0; // Not a heading
    }

    private string CleanHeadingText(string text)
    {
        // Remove numbering prefix
        return Regex.Replace(text, @"^(\d+\.)+\s*", "").Trim();
    }
}

// API Controller
[ApiController]
[Route("api/[controller]")]
public class PdfController : ControllerBase
{
    private readonly PdfParserService _pdfParser;

    public PdfController(PdfParserService pdfParser)
    {
        _pdfParser = pdfParser;
    }

    [HttpPost("parse")]
    public async Task<IActionResult> ParsePdf(IFormFile file)
    {
        if (file == null || file.Length == 0)
            return BadRequest("No file uploaded");

        var tempPath = Path.GetTempFileName();
        
        try
        {
            using (var stream = new FileStream(tempPath, FileMode.Create))
            {
                await file.CopyToAsync(stream);
            }

            var json = _pdfParser.ParsePdfToJson(tempPath);
            return Ok(json);
        }
        catch (Exception ex)
        {
            return StatusCode(500, $"Error parsing PDF: {ex.Message}");
        }
        finally
        {
            if (File.Exists(tempPath))
                File.Delete(tempPath);
        }
    }
}
