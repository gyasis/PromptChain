<script lang="ts">
  import { createEventDispatcher, onMount, afterUpdate } from 'svelte';
  import { Send, Bot, User, Loader2, Paperclip, Settings, RotateCcw, Plus } from 'lucide-svelte';
  import { Button } from "$lib/components/ui/button/index.js";
  import { Input } from "$lib/components/ui/input/index.js";
  import * as Card from "$lib/components/ui/card/index.js";
  import { Badge } from "$lib/components/ui/badge/index.js";
  import { Skeleton } from "$lib/components/ui/skeleton/index.js";

  interface ChatMessage {
    id: string;
    type: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: Date;
    metadata?: {
      sessionId?: string;
      sources?: string[];
      confidence?: number;
      processingTime?: number;
    };
  }

  interface ChatSession {
    id: string;
    title: string;
    messages: ChatMessage[];
    createdAt: Date;
    researchSessionId?: string;
  }

  export let sessionId: string | null = null;
  export let researchSessionId: string | null = null;
  export let isVisible = true;
  export let placeholder = "Ask questions about your research findings...";

  const dispatch = createEventDispatcher<{
    sendMessage: { message: string; sessionId: string; researchSessionId?: string };
    newSession: void;
    clearSession: { sessionId: string };
    selectSession: { sessionId: string };
  }>();

  // Chat state
  let currentMessage = '';
  let isLoading = false;
  let isTyping = false;
  let messages: ChatMessage[] = [];
  let chatContainer: HTMLDivElement;
  let messageInput: HTMLInputElement;

  // Session management
  let currentSessionId = sessionId || generateSessionId();
  let sessions: ChatSession[] = [];

  // Demo mode state
  let isDemoMode = true;
  let demoMessages: ChatMessage[] = [
    {
      id: 'demo-1',
      type: 'system',
      content: 'Chat interface connected. You can ask questions about your research findings.',
      timestamp: new Date(),
      metadata: { sessionId: currentSessionId }
    },
    {
      id: 'demo-2', 
      type: 'user',
      content: 'What are the main themes in the papers I found about machine learning in healthcare?',
      timestamp: new Date(Date.now() - 60000),
      metadata: { sessionId: currentSessionId }
    },
    {
      id: 'demo-3',
      type: 'assistant',
      content: `Based on your research findings, I've identified several key themes in machine learning for healthcare:

**1. Deep Learning for Medical Imaging**
- Convolutional Neural Networks for diagnosis
- Transfer learning approaches
- Transformer-based architectures

**2. Clinical Decision Support**
- Electronic health record analysis
- Predictive modeling for patient outcomes
- Real-time monitoring systems

**3. Privacy and Ethics**
- Federated learning for privacy preservation
- Explainable AI for medical decisions
- Bias detection and mitigation

**4. Multi-Modal Integration**
- Combining images, text, and structured data
- Cross-modal learning approaches
- Enhanced prediction accuracy

The papers show a strong focus on practical deployment challenges and regulatory compliance.`,
      timestamp: new Date(Date.now() - 45000),
      metadata: { 
        sessionId: currentSessionId,
        sources: ['paper-1', 'paper-2', 'paper-3'],
        confidence: 0.92,
        processingTime: 2.3
      }
    }
  ];

  $: if (isDemoMode) {
    messages = demoMessages;
  }

  function generateSessionId(): string {
    return `chat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  function generateMessageId(): string {
    return `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  async function sendMessage() {
    if (!currentMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      id: generateMessageId(),
      type: 'user',
      content: currentMessage.trim(),
      timestamp: new Date(),
      metadata: { sessionId: currentSessionId, ...(researchSessionId && { researchSessionId }) }
    };

    // Add user message to chat
    messages = [...messages, userMessage];
    const messageToSend = currentMessage.trim();
    currentMessage = '';
    isLoading = true;

    // Scroll to bottom after user message
    setTimeout(scrollToBottom, 50);

    try {
      // In demo mode, simulate AI response
      if (isDemoMode) {
        await simulateAIResponse(messageToSend);
      } else {
        // Dispatch event for real API call
        dispatch('sendMessage', {
          message: messageToSend,
          sessionId: currentSessionId,
          researchSessionId: researchSessionId || undefined
        });
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Add error message
      const errorMessage: ChatMessage = {
        id: generateMessageId(),
        type: 'system',
        content: 'Sorry, I encountered an error processing your message. Please try again.',
        timestamp: new Date(),
        metadata: { sessionId: currentSessionId }
      };
      messages = [...messages, errorMessage];
    } finally {
      isLoading = false;
      setTimeout(scrollToBottom, 100);
    }
  }

  async function simulateAIResponse(userMessage: string) {
    // Simulate typing indicator
    isTyping = true;
    await new Promise(resolve => setTimeout(resolve, 1500));
    isTyping = false;

    // Generate contextual response based on user input
    let responseContent = '';
    const lowerMessage = userMessage.toLowerCase();

    if (lowerMessage.includes('summary') || lowerMessage.includes('summarize')) {
      responseContent = `Here's a summary of your research findings:

**Key Findings:**
• 5 papers found across ArXiv, PubMed, and Sci-Hub sources
• Primary focus on AI applications in healthcare and medical imaging
• Strong emphasis on explainable AI and privacy-preserving methods
• Recent developments in transformer architectures for clinical data

**Research Gaps Identified:**
• Limited real-world deployment studies
• Need for larger multi-institutional datasets
• Regulatory framework development still in progress

**Recommended Next Steps:**
• Focus on implementation challenges
• Investigate regulatory compliance requirements
• Explore federated learning approaches`;
    } else if (lowerMessage.includes('method') || lowerMessage.includes('approach')) {
      responseContent = `The papers describe several methodological approaches:

**Deep Learning Methods:**
- Convolutional Neural Networks for medical image analysis
- Transformer architectures for clinical text processing
- Multi-modal learning combining images and structured data

**Privacy-Preserving Techniques:**
- Federated learning across medical institutions
- Differential privacy for patient data protection
- Secure multi-party computation protocols

**Evaluation Frameworks:**
- Cross-validation on clinical datasets
- External validation across different hospitals
- Fairness metrics for bias assessment`;
    } else if (lowerMessage.includes('limitation') || lowerMessage.includes('challenge')) {
      responseContent = `Key limitations and challenges identified in the research:

**Technical Challenges:**
• Data heterogeneity across medical institutions
• Limited availability of labeled clinical datasets
• Model interpretability requirements for clinical adoption
• Computational resource constraints in healthcare settings

**Regulatory & Ethical Challenges:**
• FDA approval processes for AI medical devices
• Patient consent and data privacy regulations
• Liability and accountability for AI-driven decisions
• Bias and fairness in healthcare AI systems

**Implementation Barriers:**
• Integration with existing hospital information systems
• Clinician training and acceptance
• Cost-effectiveness validation
• Maintenance and continuous learning requirements`;
    } else {
      responseContent = `I can help you analyze your research findings. Here are some things I can assist with:

**Research Analysis:**
• Summarize key findings and themes
• Identify research gaps and opportunities
• Compare methodologies across papers
• Extract statistical results and metrics

**Literature Review Support:**
• Generate structured summaries
• Create comparison tables
• Identify conflicting findings
• Suggest areas for future research

**Specific Questions I Can Answer:**
• "What methods were used in these studies?"
• "What are the main limitations mentioned?"
• "How do the results compare across papers?"
• "What datasets were used?"

Ask me anything specific about your ${messages.filter(m => m.type !== 'system').length > 1 ? 'research findings' : 'research topic'}!`;
    }

    const assistantMessage: ChatMessage = {
      id: generateMessageId(),
      type: 'assistant',
      content: responseContent,
      timestamp: new Date(),
      metadata: {
        sessionId: currentSessionId,
        sources: ['paper-1', 'paper-2', 'paper-3'],
        confidence: Math.random() * 0.3 + 0.7, // Random confidence between 0.7-1.0
        processingTime: Math.random() * 2 + 1 // Random processing time 1-3 seconds
      }
    };

    messages = [...messages, assistantMessage];
  }

  function handleKeyPress(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  }

  function scrollToBottom() {
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  function clearCurrentSession() {
    messages = [];
    dispatch('clearSession', { sessionId: currentSessionId });
  }

  function createNewSession() {
    currentSessionId = generateSessionId();
    messages = [];
    dispatch('newSession');
  }

  function formatTimestamp(timestamp: Date): string {
    return timestamp.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function getMessageIcon(type: string) {
    switch (type) {
      case 'user': return User;
      case 'assistant': return Bot;
      default: return Bot;
    }
  }

  function getMessageBgClass(type: string): string {
    switch (type) {
      case 'user': return 'bg-primary-50 border-primary-200';
      case 'assistant': return 'bg-white border-neutral-200';
      case 'system': return 'bg-neutral-50 border-neutral-200';
      default: return 'bg-white border-neutral-200';
    }
  }

  // Scroll to bottom when messages change
  afterUpdate(() => {
    scrollToBottom();
  });

  onMount(() => {
    // Focus input on mount
    if (messageInput) {
      messageInput.focus();
    }
  });
</script>

{#if isVisible}
  <div class="flex flex-col h-full bg-white">
    <!-- Chat Header -->
    <div class="border-b border-neutral-200 p-4 flex items-center justify-between bg-white">
      <div class="flex items-center gap-3">
        <div class="p-2 bg-primary-100 rounded-lg">
          <Bot class="h-5 w-5 text-primary-600" />
        </div>
        <div>
          <h3 class="font-semibold text-neutral-900">Research Assistant</h3>
          <p class="text-sm text-neutral-500">
            {researchSessionId ? `Connected to research session` : 'Ask questions about your research'}
          </p>
        </div>
      </div>
      
      <div class="flex items-center gap-2">
        {#if isDemoMode}
          <Badge variant="secondary" class="text-xs">Demo Mode</Badge>
        {/if}
        <Button variant="ghost" size="sm" on:click={createNewSession} title="New chat">
          <Plus class="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="sm" on:click={clearCurrentSession} title="Clear chat">
          <RotateCcw class="h-4 w-4" />
        </Button>
      </div>
    </div>

    <!-- Messages Container -->
    <div 
      class="flex-1 overflow-y-auto p-4 space-y-4 bg-neutral-50"
      bind:this={chatContainer}
    >
      {#each messages as message (message.id)}
        <div class="flex gap-3 {message.type === 'user' ? 'justify-end' : 'justify-start'}">
          {#if message.type !== 'user'}
            <div class="flex-shrink-0">
              <div class="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
                <svelte:component this={getMessageIcon(message.type)} class="h-4 w-4 text-primary-600" />
              </div>
            </div>
          {/if}
          
          <div class="max-w-[80%] {message.type === 'user' ? 'order-first' : ''}">
            <div class="rounded-lg p-3 border {getMessageBgClass(message.type)} shadow-sm">
              <div class="prose prose-sm max-w-none">
                {#if message.content.includes('**')}
                  <!-- Render markdown-style formatting -->
                  {@html message.content
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/^• /gm, '• ')
                    .replace(/\n/g, '<br>')
                  }
                {:else}
                  <p class="text-neutral-800 whitespace-pre-wrap">{message.content}</p>
                {/if}
              </div>
              
              <!-- Message metadata -->
              <div class="flex items-center justify-between mt-2 pt-2 border-t border-neutral-100">
                <div class="text-xs text-neutral-500">
                  {formatTimestamp(message.timestamp)}
                </div>
                
                {#if message.metadata?.confidence}
                  <div class="text-xs text-neutral-500 flex items-center gap-2">
                    <span>Confidence: {Math.round(message.metadata.confidence * 100)}%</span>
                    {#if message.metadata.processingTime}
                      <span>• {message.metadata.processingTime.toFixed(1)}s</span>
                    {/if}
                  </div>
                {/if}
              </div>
            </div>
          </div>
          
          {#if message.type === 'user'}
            <div class="flex-shrink-0">
              <div class="w-8 h-8 rounded-full bg-neutral-100 flex items-center justify-center">
                <User class="h-4 w-4 text-neutral-600" />
              </div>
            </div>
          {/if}
        </div>
      {/each}
      
      <!-- Typing Indicator -->
      {#if isTyping}
        <div class="flex gap-3 justify-start">
          <div class="flex-shrink-0">
            <div class="w-8 h-8 rounded-full bg-primary-100 flex items-center justify-center">
              <Bot class="h-4 w-4 text-primary-600" />
            </div>
          </div>
          <div class="bg-white border border-neutral-200 rounded-lg p-3 shadow-sm">
            <div class="flex items-center gap-2">
              <div class="flex gap-1">
                <div class="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style="animation-delay: 0ms"></div>
                <div class="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style="animation-delay: 150ms"></div>
                <div class="w-2 h-2 bg-neutral-400 rounded-full animate-bounce" style="animation-delay: 300ms"></div>
              </div>
              <span class="text-sm text-neutral-500">Thinking...</span>
            </div>
          </div>
        </div>
      {/if}
    </div>

    <!-- Message Input -->
    <div class="border-t border-neutral-200 p-4 bg-white">
      <div class="flex gap-3 items-end">
        <div class="flex-1">
          <Input
            bind:value={currentMessage}
            bind:this={messageInput}
            onkeydown={handleKeyPress}
            placeholder={placeholder}
            disabled={isLoading}
            class="resize-none"
          />
        </div>
        
        <Button 
          onclick={sendMessage}
          disabled={!currentMessage.trim() || isLoading}
          class="flex-shrink-0"
        >
          {#if isLoading}
            <Loader2 class="h-4 w-4 animate-spin" />
          {:else}
            <Send class="h-4 w-4" />
          {/if}
        </Button>
      </div>
      
      <div class="mt-2 text-xs text-neutral-500 flex items-center justify-between">
        <span>Press Enter to send, Shift+Enter for new line</span>
        {#if researchSessionId}
          <span class="text-primary-600">Connected to research session</span>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .prose {
    max-width: none;
  }
  
  .prose p {
    margin: 0.5em 0;
  }
  
  :global(.prose strong) {
    font-weight: 600;
    color: #1f2937;
  }
  
  :global(.prose em) {
    font-style: italic;
  }
  
  /* Custom scrollbar */
  :global(.overflow-y-auto) {
    scrollbar-width: thin;
    scrollbar-color: #d1d5db #f9fafb;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar) {
    width: 6px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-track) {
    background: #f9fafb;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb) {
    background: #d1d5db;
    border-radius: 3px;
  }
  
  :global(.overflow-y-auto::-webkit-scrollbar-thumb:hover) {
    background: #9ca3af;
  }
</style>