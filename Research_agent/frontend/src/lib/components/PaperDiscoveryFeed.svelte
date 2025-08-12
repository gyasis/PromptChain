<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { FileText, ExternalLink, Calendar, Users, BookOpen } from 'lucide-svelte';
  
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

  export let papers: Paper[] = [];
  export let sessionId: string;
  export let isVisible = true;

  const dispatch = createEventDispatcher<{
    openPdf: { paper: Paper };
    downloadPdf: { paper: Paper };
  }>();

  function handleOpenPdf(paper: Paper) {
    if (paper.status === 'ready' && paper.pdfUrl) {
      dispatch('openPdf', { paper });
    }
  }

  function handleDownloadPdf(paper: Paper) {
    if (paper.url || paper.pdfUrl) {
      dispatch('downloadPdf', { paper });
    }
  }

  function getSourceIcon(source: string): string {
    switch (source) {
      case 'arxiv': return '📄';
      case 'pubmed': return '🧬'; 
      case 'scihub': return '🔬';
      default: return '📚';
    }
  }

  function getSourceColor(source: string): string {
    switch (source) {
      case 'arxiv': return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'pubmed': return 'bg-green-100 text-green-700 border-green-200';
      case 'scihub': return 'bg-purple-100 text-purple-700 border-purple-200';
      default: return 'bg-neutral-100 text-neutral-700 border-neutral-200';
    }
  }

  function getStatusIndicator(paper: Paper): { text: string; class: string; icon: string } {
    switch (paper.status) {
      case 'found':
        return { text: 'Found', class: 'bg-blue-100 text-blue-700', icon: '🔍' };
      case 'downloading':
        return { text: 'Getting PDF', class: 'bg-amber-100 text-amber-700', icon: '⬇️' };
      case 'ready':
        return { text: 'Ready', class: 'bg-green-100 text-green-700', icon: '✅' };
      case 'error':
        return { text: 'Error', class: 'bg-red-100 text-red-700', icon: '❌' };
      default:
        return { text: 'Unknown', class: 'bg-neutral-100 text-neutral-700', icon: '❓' };
    }
  }

  function formatAuthors(authors: string[]): string {
    if (authors.length === 0) return 'Unknown authors';
    if (authors.length === 1) return authors[0];
    if (authors.length <= 3) return authors.join(', ');
    return `${authors.slice(0, 2).join(', ')} et al. (+${authors.length - 2})`;
  }

  function formatDate(dateStr?: string): string {
    if (!dateStr) return '';
    try {
      return new Date(dateStr).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short'
      });
    } catch {
      return dateStr;
    }
  }

  function truncateAbstract(abstract: string, maxLength = 200): string {
    if (abstract.length <= maxLength) return abstract;
    return abstract.substring(0, maxLength).trim() + '...';
  }

  // Animation class for newly added papers
  function getAnimationClass(index: number): string {
    return `animate-in`;
  }
</script>

{#if isVisible}
  <div class="card p-6">
    <!-- Header -->
    <div class="section-header">
      <div>
        <h3 class="section-title flex items-center gap-2">
          <BookOpen class="h-5 w-5 text-primary-500" />
          Paper Discovery
        </h3>
        <p class="section-subtitle">Papers found during research • {papers.length} total</p>
      </div>
      
      {#if papers.length > 0}
        <div class="flex items-center gap-2 text-sm text-neutral-600">
          <div class="flex items-center gap-1">
            <div class="status-dot bg-green-500"></div>
            <span>{papers.filter(p => p.status === 'ready').length} ready</span>
          </div>
          <div class="flex items-center gap-1">
            <div class="status-dot bg-amber-500 animate-pulse"></div>
            <span>{papers.filter(p => p.status === 'downloading').length} loading</span>
          </div>
        </div>
      {/if}
    </div>

    <!-- Papers Feed -->
    <div class="space-y-4 max-h-[600px] overflow-y-auto scrollbar-thin">
      {#if papers.length === 0}
        <!-- Empty State -->
        <div class="text-center py-12">
          <BookOpen class="h-12 w-12 text-neutral-400 mx-auto mb-4" />
          <h4 class="text-lg font-medium text-neutral-600 mb-2">Searching for papers...</h4>
          <p class="text-neutral-500">Papers will appear here as they're discovered</p>
        </div>
      {:else}
        {#each papers as paper, index (paper.id)}
          <div class="card-hover p-4 {getAnimationClass(index)}" style="animation-delay: {index * 100}ms">
            <!-- Paper Header -->
            <div class="flex items-start justify-between mb-3">
              <div class="flex-1 min-w-0">
                <h4 class="font-semibold text-neutral-900 line-clamp-2 mb-2">
                  {paper.title}
                </h4>
                
                <div class="flex items-center flex-wrap gap-2 mb-2">
                  <!-- Source Badge -->
                  <span class="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium border {getSourceColor(paper.source)}">
                    <span>{getSourceIcon(paper.source)}</span>
                    {paper.source.toUpperCase()}
                  </span>
                  
                  <!-- Status Badge -->
                  <span class="inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium {getStatusIndicator(paper).class}">
                    <span>{getStatusIndicator(paper).icon}</span>
                    {getStatusIndicator(paper).text}
                  </span>
                  
                  <!-- Date -->
                  {#if paper.publishedDate}
                    <span class="text-xs text-neutral-500 flex items-center gap-1">
                      <Calendar class="h-3 w-3" />
                      {formatDate(paper.publishedDate)}
                    </span>
                  {/if}
                </div>
                
                <!-- Authors -->
                <div class="flex items-center gap-1 text-sm text-neutral-600 mb-2">
                  <Users class="h-3 w-3" />
                  <span>{formatAuthors(paper.authors)}</span>
                  {#if paper.citationCount}
                    <span class="ml-2 text-xs text-neutral-500">• {paper.citationCount} citations</span>
                  {/if}
                </div>
                
                <!-- Abstract Preview -->
                {#if paper.abstract}
                  <p class="text-sm text-neutral-600 line-clamp-3 mb-3">
                    {truncateAbstract(paper.abstract)}
                  </p>
                {/if}
                
                <!-- Tags -->
                {#if paper.tags.length > 0}
                  <div class="flex flex-wrap gap-1 mb-3">
                    {#each paper.tags.slice(0, 4) as tag}
                      <span class="px-2 py-1 bg-neutral-100 text-neutral-600 text-xs rounded-md">
                        {tag}
                      </span>
                    {/each}
                    {#if paper.tags.length > 4}
                      <span class="px-2 py-1 bg-neutral-100 text-neutral-500 text-xs rounded-md">
                        +{paper.tags.length - 4} more
                      </span>
                    {/if}
                  </div>
                {/if}
              </div>
            </div>
            
            <!-- Action Buttons -->
            <div class="flex items-center justify-between">
              <div class="flex items-center gap-2">
                <!-- PDF Viewer Button -->
                <button
                  type="button"
                  class="btn-primary px-3 py-1.5 text-sm"
                  disabled={paper.status !== 'ready'}
                  on:click={() => handleOpenPdf(paper)}
                  title={paper.status === 'ready' ? 'Open PDF viewer' : 'PDF not ready yet'}
                >
                  <FileText class="h-4 w-4 mr-1" />
                  {#if paper.status === 'downloading'}
                    <div class="spinner mr-1"></div>
                    Loading...
                  {:else if paper.status === 'ready'}
                    View PDF
                  {:else if paper.status === 'error'}
                    PDF Error
                  {:else}
                    PDF Pending
                  {/if}
                </button>
                
                <!-- External Link -->
                {#if paper.url}
                  <a
                    href={paper.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    class="btn-ghost px-3 py-1.5 text-sm"
                    title="Open in new tab"
                  >
                    <ExternalLink class="h-4 w-4 mr-1" />
                    Source
                  </a>
                {/if}
              </div>
              
              <!-- Paper ID for debugging -->
              <span class="text-xs text-neutral-400 font-mono">#{paper.id.slice(-6)}</span>
            </div>
          </div>
        {/each}
      {/if}
    </div>

    <!-- Footer Stats -->
    {#if papers.length > 0}
      <div class="border-t border-neutral-200 mt-4 pt-4">
        <div class="grid grid-cols-3 gap-4 text-center text-sm">
          <div>
            <div class="font-semibold text-neutral-900">{papers.filter(p => p.source === 'arxiv').length}</div>
            <div class="text-neutral-600">ArXiv</div>
          </div>
          <div>
            <div class="font-semibold text-neutral-900">{papers.filter(p => p.source === 'pubmed').length}</div>
            <div class="text-neutral-600">PubMed</div>
          </div>
          <div>
            <div class="font-semibold text-neutral-900">{papers.filter(p => p.source === 'scihub').length}</div>
            <div class="text-neutral-600">Sci-Hub</div>
          </div>
        </div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .animate-in {
    animation: slideInUp 0.4s ease-out;
  }

  @keyframes slideInUp {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>