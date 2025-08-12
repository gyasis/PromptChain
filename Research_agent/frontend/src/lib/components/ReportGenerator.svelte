<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { FileText, Download, Eye, Settings, Calendar, BarChart3, FileImage, Copy, Check } from 'lucide-svelte';
  import { Button } from "$lib/components/ui/button/index.js";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import { Textarea } from "$lib/components/ui/textarea/index.js";
  import * as Select from "$lib/components/ui/select/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Switch } from "$lib/components/ui/switch/index.js";
  import { Separator } from "$lib/components/ui/separator/index.js";

  interface Paper {
    id: string;
    title: string;
    authors: string[];
    abstract: string;
    source: 'arxiv' | 'pubmed' | 'scihub';
    url?: string;
    pdfUrl?: string;
    publishedDate?: string;
    citationCount?: number;
    tags: string[];
    status: 'found' | 'downloading' | 'ready' | 'error';
  }

  interface ReportConfig {
    title: string;
    author: string;
    includeAbstract: boolean;
    includeCitations: boolean;
    includeVisualization: boolean;
    includeBibliography: boolean;
    includeMethodology: boolean;
    format: 'pdf' | 'word' | 'markdown' | 'html' | 'latex';
    template: 'academic' | 'business' | 'technical' | 'executive';
    customSections: string[];
  }

  export let papers: Paper[] = [];
  export let sessionId: string | null = null;
  export let isVisible = true;
  export let researchTopic = '';

  const dispatch = createEventDispatcher<{
    generateReport: { config: ReportConfig; papers: Paper[]; sessionId: string | null };
    previewReport: { config: ReportConfig; papers: Paper[] };
    exportReport: { config: ReportConfig; format: string };
  }>();

  // Report configuration state
  let reportConfig: ReportConfig = {
    title: researchTopic || 'Research Analysis Report',
    author: 'Research Agent',
    includeAbstract: true,
    includeCitations: true,
    includeVisualization: true,
    includeBibliography: true,
    includeMethodology: true,
    format: 'pdf',
    template: 'academic',
    customSections: []
  };

  let isGenerating = false;
  let showPreview = false;
  let generatedReport = '';
  let copied = false;

  // Report templates with descriptions
  const templates = [
    { 
      value: 'academic', 
      label: 'Academic Paper',
      description: 'Formal academic structure with abstract, methodology, results, and bibliography'
    },
    { 
      value: 'business', 
      label: 'Business Report',
      description: 'Executive summary focused with key insights and actionable recommendations'
    },
    { 
      value: 'technical', 
      label: 'Technical Documentation',
      description: 'Detailed technical analysis with methodology and implementation details'
    },
    { 
      value: 'executive', 
      label: 'Executive Summary',
      description: 'Concise high-level overview with key findings and strategic implications'
    }
  ];

  // Export formats with descriptions
  const formats = [
    { value: 'pdf', label: 'PDF', icon: '📄', description: 'Professional document format' },
    { value: 'word', label: 'Word', icon: '📝', description: 'Microsoft Word document' },
    { value: 'markdown', label: 'Markdown', icon: '📋', description: 'Plain text with formatting' },
    { value: 'html', label: 'HTML', icon: '🌐', description: 'Web page format' },
    { value: 'latex', label: 'LaTeX', icon: '📑', description: 'Academic typesetting format' }
  ];

  $: if (researchTopic && !reportConfig.title.includes(researchTopic)) {
    reportConfig.title = `${researchTopic} - Research Analysis Report`;
  }

  async function generateReport() {
    if (papers.length === 0) {
      alert('No papers available to generate report. Please complete a research session first.');
      return;
    }

    isGenerating = true;
    try {
      // Simulate report generation - in production this would call the backend API
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      generatedReport = generateMockReport(reportConfig, papers);
      showPreview = true;
      
      dispatch('generateReport', {
        config: reportConfig,
        papers,
        sessionId
      });
    } catch (error) {
      console.error('Failed to generate report:', error);
      alert('Failed to generate report. Please try again.');
    } finally {
      isGenerating = false;
    }
  }

  function generateMockReport(config: ReportConfig, papers: Paper[]): string {
    const sections = [];

    // Title and metadata
    sections.push(`# ${config.title}\n\n**Author:** ${config.author}\n**Generated:** ${new Date().toLocaleDateString()}\n**Papers Analyzed:** ${papers.length}\n\n`);

    // Abstract
    if (config.includeAbstract) {
      sections.push(`## Abstract\n\nThis research analysis examines ${papers.length} papers related to "${researchTopic}". The study synthesizes findings from multiple sources including ArXiv, PubMed, and academic databases to provide comprehensive insights into current research trends and methodologies.\n\n`);
    }

    // Methodology
    if (config.includeMethodology) {
      sections.push(`## Methodology\n\n### Literature Search Strategy\n- **Sources:** ArXiv, PubMed, Sci-Hub\n- **Search Terms:** ${researchTopic}\n- **Papers Retrieved:** ${papers.length}\n- **Analysis Framework:** 3-tier RAG system (LightRAG → PaperQA2 → GraphRAG)\n\n### Quality Assessment\n- Citation analysis and impact metrics\n- Author expertise evaluation\n- Publication recency assessment\n\n`);
    }

    // Key Findings
    sections.push(`## Key Findings\n\n`);

    // Paper analysis
    const totalCitations = papers.reduce((sum, p) => sum + (p.citationCount || 0), 0);
    const uniqueAuthors = new Set(papers.flatMap(p => p.authors || [])).size;
    const sourceDist = papers.reduce((acc, p) => {
      acc[p.source] = (acc[p.source] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    sections.push(`### Research Landscape Overview\n- **Total Papers:** ${papers.length}\n- **Total Citations:** ${totalCitations.toLocaleString()}\n- **Unique Authors:** ${uniqueAuthors}\n- **Average Citations per Paper:** ${Math.round(totalCitations / papers.length)}\n\n`);

    sections.push(`### Source Distribution\n`);
    Object.entries(sourceDist).forEach(([source, count]) => {
      const percentage = Math.round((count / papers.length) * 100);
      sections.push(`- **${source.toUpperCase()}:** ${count} papers (${percentage}%)\n`);
    });
    sections.push('\n');

    // Top papers
    const topPapers = papers
      .filter(p => p.citationCount && p.citationCount > 0)
      .sort((a, b) => (b.citationCount || 0) - (a.citationCount || 0))
      .slice(0, 5);

    if (topPapers.length > 0) {
      sections.push(`### Most Cited Papers\n\n`);
      topPapers.forEach((paper, index) => {
        sections.push(`${index + 1}. **${paper.title}**\n   - Authors: ${paper.authors.slice(0, 3).join(', ')}${paper.authors.length > 3 ? ' et al.' : ''}\n   - Citations: ${paper.citationCount}\n   - Source: ${paper.source.toUpperCase()}\n\n`);
      });
    }

    // Research themes
    const allTags = papers.flatMap(p => p.tags || []);
    const tagCounts = allTags.reduce((acc, tag) => {
      acc[tag] = (acc[tag] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const topTags = Object.entries(tagCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);

    if (topTags.length > 0) {
      sections.push(`### Research Themes\n\nThe most prominent research themes identified:\n\n`);
      topTags.forEach(([tag, count]) => {
        sections.push(`- **${tag}** (${count} papers)\n`);
      });
      sections.push('\n');
    }

    // Conclusions
    sections.push(`## Conclusions\n\nThis analysis of ${papers.length} papers provides comprehensive insights into the current state of research in "${researchTopic}". Key trends include:\n\n1. **Methodological Diversity:** Research spans multiple approaches and methodologies\n2. **Citation Impact:** Average impact of ${Math.round(totalCitations / papers.length)} citations per paper\n3. **Source Diversity:** Papers drawn from ${Object.keys(sourceDist).length} different academic sources\n4. **Research Maturity:** Evidence of established research community with ${uniqueAuthors} unique contributors\n\n`);

    // Bibliography
    if (config.includeBibliography) {
      sections.push(`## Bibliography\n\n`);
      papers.forEach((paper, index) => {
        const authors = paper.authors.length > 0 ? paper.authors.join(', ') : 'Unknown authors';
        const year = paper.publishedDate ? new Date(paper.publishedDate).getFullYear() : 'n.d.';
        sections.push(`${index + 1}. ${authors}. (${year}). ${paper.title}. Retrieved from ${paper.url || 'Academic database'}\n\n`);
      });
    }

    return sections.join('');
  }

  async function previewReport() {
    showPreview = true;
    if (!generatedReport) {
      generatedReport = generateMockReport(reportConfig, papers);
    }
    dispatch('previewReport', { config: reportConfig, papers });
  }

  async function exportReport(format: string) {
    if (!generatedReport) {
      await generateReport();
    }
    
    dispatch('exportReport', { config: reportConfig, format });
    
    // Simulate file download
    const filename = `research-report-${sessionId || Date.now()}.${format}`;
    console.log(`Exporting report as ${filename}`);
    
    // In production, this would trigger actual file download
    if (format === 'markdown') {
      downloadAsFile(generatedReport, filename, 'text/markdown');
    }
  }

  function downloadAsFile(content: string, filename: string, mimeType: string) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  async function copyToClipboard() {
    if (generatedReport) {
      try {
        await navigator.clipboard.writeText(generatedReport);
        copied = true;
        setTimeout(() => copied = false, 2000);
      } catch (err) {
        console.error('Failed to copy to clipboard:', err);
      }
    }
  }

  function addCustomSection() {
    const sectionName = prompt('Enter custom section name:');
    if (sectionName && !reportConfig.customSections.includes(sectionName)) {
      reportConfig.customSections = [...reportConfig.customSections, sectionName];
    }
  }

  function removeCustomSection(index: number) {
    reportConfig.customSections = reportConfig.customSections.filter((_, i) => i !== index);
  }
</script>

{#if isVisible}
  <div class="space-y-6">
    <!-- Header -->
    <div class="section-header">
      <div>
        <h3 class="section-title flex items-center gap-2">
          <FileText class="h-5 w-5 text-primary-500" />
          Research Report Generator
        </h3>
        <p class="section-subtitle">Generate comprehensive reports from your research findings</p>
      </div>
      
      <div class="flex gap-2">
        <Button variant="outline" onclick={previewReport} disabled={papers.length === 0}>
          <Eye class="h-4 w-4 mr-2" />
          Preview
        </Button>
        <Button onclick={generateReport} disabled={isGenerating || papers.length === 0}>
          {#if isGenerating}
            <div class="spinner mr-2"></div>
            Generating...
          {:else}
            <FileText class="h-4 w-4 mr-2" />
            Generate Report
          {/if}
        </Button>
      </div>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Configuration Panel -->
      <div class="space-y-6">
        <!-- Basic Settings -->
        <Card.Root>
          <Card.Header>
            <Card.Title class="flex items-center gap-2">
              <Settings class="h-4 w-4" />
              Report Configuration
            </Card.Title>
          </Card.Header>
          <Card.Content class="space-y-4">
            <!-- Title -->
            <div>
              <label class="block text-sm font-medium text-neutral-700 mb-2">Report Title</label>
              <Input bind:value={reportConfig.title} placeholder="Enter report title..." />
            </div>

            <!-- Author -->
            <div>
              <label class="block text-sm font-medium text-neutral-700 mb-2">Author</label>
              <Input bind:value={reportConfig.author} placeholder="Report author..." />
            </div>

            <!-- Template Selection -->
            <div>
              <label class="block text-sm font-medium text-neutral-700 mb-2">Report Template</label>
              <Select.Root bind:selected={reportConfig.template}>
                <Select.Trigger>
                  <Select.Value placeholder="Select template..." />
                </Select.Trigger>
                <Select.Content>
                  {#each templates as template}
                    <Select.Item value={template.value}>
                      <div>
                        <div class="font-medium">{template.label}</div>
                        <div class="text-xs text-neutral-500">{template.description}</div>
                      </div>
                    </Select.Item>
                  {/each}
                </Select.Content>
              </Select.Root>
            </div>

            <!-- Export Format -->
            <div>
              <label class="block text-sm font-medium text-neutral-700 mb-2">Export Format</label>
              <div class="grid grid-cols-3 gap-2">
                {#each formats as format}
                  <button
                    type="button"
                    class="p-3 border rounded-lg text-center hover:bg-neutral-50 transition-colors {reportConfig.format === format.value ? 'border-primary-500 bg-primary-50' : 'border-neutral-200'}"
                    on:click={() => reportConfig.format = format.value}
                  >
                    <div class="text-lg mb-1">{format.icon}</div>
                    <div class="text-xs font-medium">{format.label}</div>
                  </button>
                {/each}
              </div>
            </div>
          </Card.Content>
        </Card.Root>

        <!-- Content Options -->
        <Card.Root>
          <Card.Header>
            <Card.Title>Content Sections</Card.Title>
            <Card.Description>Choose what to include in your report</Card.Description>
          </Card.Header>
          <Card.Content class="space-y-4">
            <div class="grid grid-cols-1 gap-4">
              <!-- Include toggles -->
              <div class="flex items-center justify-between">
                <label class="text-sm font-medium">Abstract & Summary</label>
                <Switch bind:checked={reportConfig.includeAbstract} />
              </div>
              
              <div class="flex items-center justify-between">
                <label class="text-sm font-medium">Methodology</label>
                <Switch bind:checked={reportConfig.includeMethodology} />
              </div>
              
              <div class="flex items-center justify-between">
                <label class="text-sm font-medium">Citations & References</label>
                <Switch bind:checked={reportConfig.includeCitations} />
              </div>
              
              <div class="flex items-center justify-between">
                <label class="text-sm font-medium">Data Visualizations</label>
                <Switch bind:checked={reportConfig.includeVisualization} />
              </div>
              
              <div class="flex items-center justify-between">
                <label class="text-sm font-medium">Bibliography</label>
                <Switch bind:checked={reportConfig.includeBibliography} />
              </div>
            </div>

            <Separator />

            <!-- Custom Sections -->
            <div>
              <div class="flex items-center justify-between mb-2">
                <label class="text-sm font-medium">Custom Sections</label>
                <Button variant="ghost" size="sm" onclick={addCustomSection}>
                  <Plus class="h-3 w-3 mr-1" />
                  Add
                </Button>
              </div>
              
              {#if reportConfig.customSections.length > 0}
                <div class="space-y-2">
                  {#each reportConfig.customSections as section, index}
                    <div class="flex items-center justify-between p-2 bg-neutral-50 rounded">
                      <span class="text-sm">{section}</span>
                      <Button variant="ghost" size="sm" onclick={() => removeCustomSection(index)}>
                        ×
                      </Button>
                    </div>
                  {/each}
                </div>
              {:else}
                <p class="text-sm text-neutral-500">No custom sections added</p>
              {/if}
            </div>
          </Card.Content>
        </Card.Root>
      </div>

      <!-- Preview/Export Panel -->
      <div class="space-y-6">
        <!-- Data Overview -->
        <Card.Root>
          <Card.Header>
            <Card.Title class="flex items-center gap-2">
              <BarChart3 class="h-4 w-4" />
              Data Overview
            </Card.Title>
          </Card.Header>
          <Card.Content>
            <div class="grid grid-cols-2 gap-4 text-center">
              <div>
                <div class="text-2xl font-bold text-primary-600">{papers.length}</div>
                <div class="text-sm text-neutral-500">Papers</div>
              </div>
              <div>
                <div class="text-2xl font-bold text-green-600">
                  {papers.reduce((sum, p) => sum + (p.citationCount || 0), 0)}
                </div>
                <div class="text-sm text-neutral-500">Citations</div>
              </div>
              <div>
                <div class="text-2xl font-bold text-blue-600">
                  {new Set(papers.flatMap(p => p.authors || [])).size}
                </div>
                <div class="text-sm text-neutral-500">Authors</div>
              </div>
              <div>
                <div class="text-2xl font-bold text-purple-600">
                  {new Set(papers.map(p => p.source)).size}
                </div>
                <div class="text-sm text-neutral-500">Sources</div>
              </div>
            </div>

            {#if papers.length === 0}
              <div class="mt-4 p-4 bg-amber-50 border border-amber-200 rounded-lg">
                <p class="text-sm text-amber-700">
                  No papers available. Please complete a research session first to generate a report.
                </p>
              </div>
            {/if}
          </Card.Content>
        </Card.Root>

        <!-- Export Options -->
        <Card.Root>
          <Card.Header>
            <Card.Title class="flex items-center gap-2">
              <Download class="h-4 w-4" />
              Export Options
            </Card.Title>
          </Card.Header>
          <Card.Content>
            <div class="grid grid-cols-2 gap-3">
              {#each formats as format}
                <Button
                  variant="outline"
                  class="h-auto p-3 flex flex-col items-center gap-2"
                  disabled={papers.length === 0}
                  onclick={() => exportReport(format.value)}
                >
                  <span class="text-lg">{format.icon}</span>
                  <span class="text-xs font-medium">{format.label}</span>
                </Button>
              {/each}
            </div>
            
            {#if generatedReport}
              <Separator class="my-4" />
              <Button
                variant="ghost"
                class="w-full"
                onclick={copyToClipboard}
              >
                {#if copied}
                  <Check class="h-4 w-4 mr-2 text-green-600" />
                  Copied!
                {:else}
                  <Copy class="h-4 w-4 mr-2" />
                  Copy to Clipboard
                {/if}
              </Button>
            {/if}
          </Card.Content>
        </Card.Root>
      </div>
    </div>

    <!-- Preview Modal -->
    {#if showPreview && generatedReport}
      <Card.Root class="mt-6">
        <Card.Header>
          <div class="flex items-center justify-between">
            <Card.Title class="flex items-center gap-2">
              <Eye class="h-4 w-4" />
              Report Preview
            </Card.Title>
            <Button variant="ghost" onclick={() => showPreview = false}>×</Button>
          </div>
        </Card.Header>
        <Card.Content>
          <div class="bg-white border rounded-lg p-6 max-h-96 overflow-y-auto">
            <pre class="whitespace-pre-wrap text-sm font-mono">{generatedReport}</pre>
          </div>
        </Card.Content>
      </Card.Root>
    {/if}
  </div>
{/if}

<style>
  .spinner {
    width: 1rem;
    height: 1rem;
    border: 2px solid #e5e7eb;
    border-top: 2px solid #3b82f6;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
</style>