<script lang="ts">
  import { onMount } from 'svelte';
  import { BarChart3, PieChart, TrendingUp, Network, Calendar, FileText, Users, Tags } from 'lucide-svelte';
  import * as Card from "$lib/components/ui/card/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";

  interface Paper {
    id: string;
    title: string;
    authors: string[];
    source: 'arxiv' | 'pubmed' | 'scihub';
    publishedDate?: string;
    citationCount?: number;
    tags: string[];
  }

  interface VisualizationData {
    papers: Paper[];
    sessionStats: {
      totalPapers: number;
      totalAuthors: number;
      totalCitations: number;
      avgPublicationYear: number;
      timeRange: { start: string; end: string };
    };
    sourceDistribution: { [key: string]: number };
    yearDistribution: { [key: string]: number };
    authorNetwork: { [key: string]: number };
    topTags: { tag: string; count: number }[];
    citationStats: { mean: number; median: number; max: number; min: number };
  }

  export let papers: Paper[] = [];
  export let isVisible = true;
  export let title = "Research Analytics";

  let visualizationData: VisualizationData | null = null;
  let selectedVisualization = 'overview';

  // Demo data for visualization
  const demoData: VisualizationData = {
    papers: [],
    sessionStats: {
      totalPapers: 5,
      totalAuthors: 18,
      totalCitations: 669,
      avgPublicationYear: 2023,
      timeRange: { start: '2023-01-08', end: '2023-04-10' }
    },
    sourceDistribution: {
      'arxiv': 2,
      'pubmed': 2, 
      'scihub': 1
    },
    yearDistribution: {
      '2023': 5
    },
    authorNetwork: {
      'Smith, J.A.': 3,
      'Chen, L.': 2,
      'Zhang, Y.': 2,
      'Thompson, A.K.': 1,
      'Rodriguez, M.A.': 1
    },
    topTags: [
      { tag: 'deep learning', count: 3 },
      { tag: 'medical imaging', count: 2 },
      { tag: 'healthcare', count: 2 },
      { tag: 'transformers', count: 2 },
      { tag: 'privacy', count: 2 },
      { tag: 'federated learning', count: 1 },
      { tag: 'explainable AI', count: 1 },
      { tag: 'multi-modal learning', count: 1 }
    ],
    citationStats: {
      mean: 133.8,
      median: 127,
      max: 203,
      min: 89
    }
  };

  $: {
    if (papers.length > 0) {
      visualizationData = generateVisualizationData(papers);
    } else {
      visualizationData = demoData;
    }
  }

  function generateVisualizationData(papers: Paper[]): VisualizationData {
    const sourceDistribution: { [key: string]: number } = {};
    const yearDistribution: { [key: string]: number } = {};
    const authorNetwork: { [key: string]: number } = {};
    const tagCounts: { [key: string]: number } = {};

    let totalCitations = 0;
    let citationCounts: number[] = [];
    let years: number[] = [];
    let allAuthors = new Set<string>();

    papers.forEach(paper => {
      // Source distribution
      sourceDistribution[paper.source] = (sourceDistribution[paper.source] || 0) + 1;

      // Year distribution
      if (paper.publishedDate) {
        const year = new Date(paper.publishedDate).getFullYear().toString();
        yearDistribution[year] = (yearDistribution[year] || 0) + 1;
        years.push(new Date(paper.publishedDate).getFullYear());
      }

      // Author network
      paper.authors.forEach(author => {
        allAuthors.add(author);
        authorNetwork[author] = (authorNetwork[author] || 0) + 1;
      });

      // Tag analysis
      paper.tags.forEach(tag => {
        tagCounts[tag] = (tagCounts[tag] || 0) + 1;
      });

      // Citation stats
      if (paper.citationCount) {
        totalCitations += paper.citationCount;
        citationCounts.push(paper.citationCount);
      }
    });

    // Calculate citation statistics
    citationCounts.sort((a, b) => a - b);
    const citationStats = {
      mean: citationCounts.length > 0 ? totalCitations / citationCounts.length : 0,
      median: citationCounts.length > 0 ? citationCounts[Math.floor(citationCounts.length / 2)] : 0,
      max: citationCounts.length > 0 ? Math.max(...citationCounts) : 0,
      min: citationCounts.length > 0 ? Math.min(...citationCounts) : 0
    };

    // Top tags
    const topTags = Object.entries(tagCounts)
      .map(([tag, count]) => ({ tag, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    // Time range
    const sortedYears = years.sort();
    const timeRange = {
      start: papers.find(p => p.publishedDate)?.publishedDate || '',
      end: papers.filter(p => p.publishedDate).sort((a, b) => 
        new Date(b.publishedDate!).getTime() - new Date(a.publishedDate!).getTime()
      )[0]?.publishedDate || ''
    };

    return {
      papers,
      sessionStats: {
        totalPapers: papers.length,
        totalAuthors: allAuthors.size,
        totalCitations,
        avgPublicationYear: years.length > 0 ? Math.round(years.reduce((a, b) => a + b, 0) / years.length) : new Date().getFullYear(),
        timeRange
      },
      sourceDistribution,
      yearDistribution,
      authorNetwork,
      topTags,
      citationStats
    };
  }

  function getSourceColor(source: string): string {
    switch (source) {
      case 'arxiv': return 'bg-blue-500';
      case 'pubmed': return 'bg-green-500';
      case 'scihub': return 'bg-purple-500';
      default: return 'bg-neutral-500';
    }
  }

  function formatNumber(num: number): string {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  }

  function calculatePercentage(value: number, total: number): number {
    return Math.round((value / total) * 100);
  }
</script>

{#if isVisible && visualizationData}
  <div class="space-y-6">
    <!-- Header -->
    <div class="section-header">
      <div>
        <h3 class="section-title flex items-center gap-2">
          <BarChart3 class="h-5 w-5 text-primary-500" />
          {title}
        </h3>
        <p class="section-subtitle">Visual insights from your research findings</p>
      </div>
      
      <!-- Visualization Type Selector -->
      <div class="flex gap-2">
        <button 
          class="btn-ghost px-3 py-1.5 text-sm {selectedVisualization === 'overview' ? 'bg-primary-100 text-primary-700' : ''}"
          on:click={() => selectedVisualization = 'overview'}
        >
          Overview
        </button>
        <button 
          class="btn-ghost px-3 py-1.5 text-sm {selectedVisualization === 'sources' ? 'bg-primary-100 text-primary-700' : ''}"
          on:click={() => selectedVisualization = 'sources'}
        >
          Sources
        </button>
        <button 
          class="btn-ghost px-3 py-1.5 text-sm {selectedVisualization === 'trends' ? 'bg-primary-100 text-primary-700' : ''}"
          on:click={() => selectedVisualization = 'trends'}
        >
          Trends
        </button>
        <button 
          class="btn-ghost px-3 py-1.5 text-sm {selectedVisualization === 'network' ? 'bg-primary-100 text-primary-700' : ''}"
          on:click={() => selectedVisualization = 'network'}
        >
          Authors
        </button>
      </div>
    </div>

    <!-- Overview Dashboard -->
    {#if selectedVisualization === 'overview'}
      <!-- Key Metrics -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card.Root class="p-4 text-center">
          <Card.Content class="p-0">
            <div class="text-2xl font-bold text-primary-600">{visualizationData.sessionStats.totalPapers}</div>
            <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Papers Found</div>
            <div class="flex items-center justify-center gap-1 mt-2">
              <FileText class="h-3 w-3 text-neutral-400" />
              <span class="text-xs text-neutral-600">Total corpus</span>
            </div>
          </Card.Content>
        </Card.Root>

        <Card.Root class="p-4 text-center">
          <Card.Content class="p-0">
            <div class="text-2xl font-bold text-green-600">{formatNumber(visualizationData.sessionStats.totalCitations)}</div>
            <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Citations</div>
            <div class="flex items-center justify-center gap-1 mt-2">
              <TrendingUp class="h-3 w-3 text-neutral-400" />
              <span class="text-xs text-neutral-600">Impact score</span>
            </div>
          </Card.Content>
        </Card.Root>

        <Card.Root class="p-4 text-center">
          <Card.Content class="p-0">
            <div class="text-2xl font-bold text-blue-600">{visualizationData.sessionStats.totalAuthors}</div>
            <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Authors</div>
            <div class="flex items-center justify-center gap-1 mt-2">
              <Users class="h-3 w-3 text-neutral-400" />
              <span class="text-xs text-neutral-600">Unique researchers</span>
            </div>
          </Card.Content>
        </Card.Root>

        <Card.Root class="p-4 text-center">
          <Card.Content class="p-0">
            <div class="text-2xl font-bold text-purple-600">{visualizationData.sessionStats.avgPublicationYear}</div>
            <div class="text-xs text-neutral-500 uppercase tracking-wider mt-1">Avg Year</div>
            <div class="flex items-center justify-center gap-1 mt-2">
              <Calendar class="h-3 w-3 text-neutral-400" />
              <span class="text-xs text-neutral-600">Publication date</span>
            </div>
          </Card.Content>
        </Card.Root>
      </div>

      <!-- Top Tags Cloud -->
      <Card.Root class="p-6">
        <Card.Header class="pb-4">
          <Card.Title class="flex items-center gap-2">
            <Tags class="h-5 w-5" />
            Research Topics
          </Card.Title>
          <Card.Description>Most frequent keywords and topics</Card.Description>
        </Card.Header>
        <Card.Content>
          <div class="flex flex-wrap gap-2">
            {#each visualizationData.topTags as {tag, count}}
              <div class="inline-flex items-center gap-1 px-3 py-1.5 bg-neutral-100 rounded-full">
                <span class="text-sm font-medium text-neutral-700">{tag}</span>
                <Badge variant="secondary" class="text-xs h-5 min-w-[1.5rem] px-1.5">{count}</Badge>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>

      <!-- Citation Statistics -->
      <Card.Root class="p-6">
        <Card.Header class="pb-4">
          <Card.Title class="flex items-center gap-2">
            <TrendingUp class="h-5 w-5" />
            Citation Analysis
          </Card.Title>
          <Card.Description>Statistical overview of paper impact</Card.Description>
        </Card.Header>
        <Card.Content>
          <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div class="text-center p-3 bg-green-50 rounded-lg">
              <div class="text-lg font-semibold text-green-700">{Math.round(visualizationData.citationStats.mean)}</div>
              <div class="text-xs text-green-600">Mean</div>
            </div>
            <div class="text-center p-3 bg-blue-50 rounded-lg">
              <div class="text-lg font-semibold text-blue-700">{visualizationData.citationStats.median}</div>
              <div class="text-xs text-blue-600">Median</div>
            </div>
            <div class="text-center p-3 bg-purple-50 rounded-lg">
              <div class="text-lg font-semibold text-purple-700">{visualizationData.citationStats.max}</div>
              <div class="text-xs text-purple-600">Maximum</div>
            </div>
            <div class="text-center p-3 bg-orange-50 rounded-lg">
              <div class="text-lg font-semibold text-orange-700">{visualizationData.citationStats.min}</div>
              <div class="text-xs text-orange-600">Minimum</div>
            </div>
          </div>
        </Card.Content>
      </Card.Root>
    {/if}

    <!-- Source Distribution -->
    {#if selectedVisualization === 'sources'}
      <Card.Root class="p-6">
        <Card.Header class="pb-4">
          <Card.Title class="flex items-center gap-2">
            <PieChart class="h-5 w-5" />
            Source Distribution
          </Card.Title>
          <Card.Description>Papers by literature source</Card.Description>
        </Card.Header>
        <Card.Content>
          <div class="space-y-4">
            {#each Object.entries(visualizationData.sourceDistribution) as [source, count]}
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <div class="w-4 h-4 rounded {getSourceColor(source)}"></div>
                  <span class="font-medium capitalize">{source}</span>
                  <Badge variant="outline">{count} paper{count !== 1 ? 's' : ''}</Badge>
                </div>
                <div class="flex items-center gap-2">
                  <div class="w-24 bg-neutral-200 rounded-full h-2 overflow-hidden">
                    <div 
                      class="h-full {getSourceColor(source)} transition-all duration-500"
                      style="width: {calculatePercentage(count, visualizationData.sessionStats.totalPapers)}%"
                    ></div>
                  </div>
                  <span class="text-sm text-neutral-600 min-w-[3rem] text-right">
                    {calculatePercentage(count, visualizationData.sessionStats.totalPapers)}%
                  </span>
                </div>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    {/if}

    <!-- Trends Analysis -->
    {#if selectedVisualization === 'trends'}
      <Card.Root class="p-6">
        <Card.Header class="pb-4">
          <Card.Title class="flex items-center gap-2">
            <Calendar class="h-5 w-5" />
            Publication Timeline
          </Card.Title>
          <Card.Description>Papers by publication year</Card.Description>
        </Card.Header>
        <Card.Content>
          <div class="space-y-4">
            {#each Object.entries(visualizationData.yearDistribution).sort((a, b) => b[0].localeCompare(a[0])) as [year, count]}
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <span class="font-medium min-w-[4rem]">{year}</span>
                  <Badge variant="outline">{count} paper{count !== 1 ? 's' : ''}</Badge>
                </div>
                <div class="flex items-center gap-2">
                  <div class="w-32 bg-neutral-200 rounded-full h-3 overflow-hidden">
                    <div 
                      class="h-full bg-primary-500 transition-all duration-500"
                      style="width: {calculatePercentage(count, visualizationData.sessionStats.totalPapers)}%"
                    ></div>
                  </div>
                  <span class="text-sm text-neutral-600 min-w-[3rem] text-right">
                    {calculatePercentage(count, visualizationData.sessionStats.totalPapers)}%
                  </span>
                </div>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    {/if}

    <!-- Author Network -->
    {#if selectedVisualization === 'network'}
      <Card.Root class="p-6">
        <Card.Header class="pb-4">
          <Card.Title class="flex items-center gap-2">
            <Network class="h-5 w-5" />
            Author Collaboration
          </Card.Title>
          <Card.Description>Top contributors and collaboration patterns</Card.Description>
        </Card.Header>
        <Card.Content>
          <div class="space-y-3">
            {#each Object.entries(visualizationData.authorNetwork).sort((a, b) => b[1] - a[1]).slice(0, 10) as [author, count], index}
              <div class="flex items-center justify-between py-2 {index === 0 ? 'border-b border-primary-200 pb-3' : ''}">
                <div class="flex items-center gap-3">
                  <div class="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center text-white text-sm font-medium">
                    {author.split(' ').map(n => n[0]).join('').slice(0, 2)}
                  </div>
                  <div>
                    <div class="font-medium text-neutral-900">{author}</div>
                    <div class="text-xs text-neutral-500">{count} publication{count !== 1 ? 's' : ''}</div>
                  </div>
                </div>
                <div class="flex items-center gap-2">
                  {#if index === 0}
                    <Badge variant="default" class="bg-gold-100 text-gold-700">Lead Author</Badge>
                  {:else if index < 3}
                    <Badge variant="secondary">Top Contributor</Badge>
                  {/if}
                  <div class="w-16 bg-neutral-200 rounded-full h-2 overflow-hidden">
                    <div 
                      class="h-full bg-gradient-to-r from-primary-500 to-primary-600 transition-all duration-500"
                      style="width: {calculatePercentage(count, Math.max(...Object.values(visualizationData.authorNetwork)))}%"
                    ></div>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        </Card.Content>
      </Card.Root>
    {/if}
  </div>
{/if}

<style>
  .bg-gold-100 {
    background-color: #fef3c7;
  }
  
  .text-gold-700 {
    color: #b45309;
  }
</style>