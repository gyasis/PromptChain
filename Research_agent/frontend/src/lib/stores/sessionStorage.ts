// Browser detection for Vite + Svelte (not SvelteKit)
const browser = typeof window !== 'undefined';

/**
 * Session storage utilities for persisting Research Agent state
 * Handles serialization/deserialization of complex objects with proper error handling
 */

interface StorageConfig {
  prefix: string;
  expiration?: number; // milliseconds
  compression?: boolean;
}

class SessionStorageManager {
  private prefix: string;
  private defaultExpiration: number;

  constructor(config: StorageConfig = { prefix: 'research_agent_' }) {
    this.prefix = config.prefix;
    this.defaultExpiration = config.expiration || 24 * 60 * 60 * 1000; // 24 hours default
  }

  /**
   * Store data with expiration and compression support
   */
  setItem<T>(key: string, value: T, expiration?: number): boolean {
    if (!browser) return false;

    try {
      const fullKey = `${this.prefix}${key}`;
      const expirationTime = Date.now() + (expiration || this.defaultExpiration);
      
      const storageItem = {
        value,
        expiration: expirationTime,
        timestamp: Date.now(),
        version: '1.0'
      };

      const serialized = JSON.stringify(storageItem);
      
      // Check if we're approaching localStorage limits
      if (serialized.length > 1024 * 1024) { // 1MB warning
        console.warn(`Large data being stored (${Math.round(serialized.length / 1024)}KB):`, key);
      }

      localStorage.setItem(fullKey, serialized);
      return true;
    } catch (error) {
      console.error('Failed to store session data:', key, error);
      
      // If quota exceeded, try to clear expired items
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        this.clearExpired();
        try {
          const fullKey = `${this.prefix}${key}`;
          const storageItem = { value, expiration: Date.now() + (expiration || this.defaultExpiration), timestamp: Date.now() };
          localStorage.setItem(fullKey, JSON.stringify(storageItem));
          return true;
        } catch (retryError) {
          console.error('Failed to store after cleanup:', retryError);
        }
      }
      return false;
    }
  }

  /**
   * Retrieve data with automatic expiration checking
   */
  getItem<T>(key: string, defaultValue?: T): T | null {
    if (!browser) return defaultValue || null;

    try {
      const fullKey = `${this.prefix}${key}`;
      const stored = localStorage.getItem(fullKey);
      
      if (!stored) return defaultValue || null;

      const storageItem = JSON.parse(stored);
      
      // Check expiration
      if (storageItem.expiration && Date.now() > storageItem.expiration) {
        this.removeItem(key);
        return defaultValue || null;
      }

      return storageItem.value as T;
    } catch (error) {
      console.error('Failed to retrieve session data:', key, error);
      // Remove corrupted data
      this.removeItem(key);
      return defaultValue || null;
    }
  }

  /**
   * Remove item from storage
   */
  removeItem(key: string): boolean {
    if (!browser) return false;

    try {
      const fullKey = `${this.prefix}${key}`;
      localStorage.removeItem(fullKey);
      return true;
    } catch (error) {
      console.error('Failed to remove session data:', key, error);
      return false;
    }
  }

  /**
   * Clear all items with this prefix
   */
  clear(): number {
    if (!browser) return 0;

    let cleared = 0;
    try {
      const keys = Object.keys(localStorage).filter(k => k.startsWith(this.prefix));
      keys.forEach(key => {
        localStorage.removeItem(key);
        cleared++;
      });
    } catch (error) {
      console.error('Failed to clear session data:', error);
    }
    return cleared;
  }

  /**
   * Clear expired items only
   */
  clearExpired(): number {
    if (!browser) return 0;

    let cleared = 0;
    try {
      const keys = Object.keys(localStorage).filter(k => k.startsWith(this.prefix));
      
      keys.forEach(fullKey => {
        try {
          const stored = localStorage.getItem(fullKey);
          if (stored) {
            const storageItem = JSON.parse(stored);
            if (storageItem.expiration && Date.now() > storageItem.expiration) {
              localStorage.removeItem(fullKey);
              cleared++;
            }
          }
        } catch (error) {
          // Remove corrupted items
          localStorage.removeItem(fullKey);
          cleared++;
        }
      });
    } catch (error) {
      console.error('Failed to clear expired session data:', error);
    }
    return cleared;
  }

  /**
   * Get storage stats
   */
  getStats(): { totalItems: number; totalSize: number; expiredItems: number } {
    if (!browser) return { totalItems: 0, totalSize: 0, expiredItems: 0 };

    let totalItems = 0;
    let totalSize = 0;
    let expiredItems = 0;

    try {
      const keys = Object.keys(localStorage).filter(k => k.startsWith(this.prefix));
      
      keys.forEach(fullKey => {
        try {
          const stored = localStorage.getItem(fullKey);
          if (stored) {
            totalItems++;
            totalSize += stored.length;
            
            const storageItem = JSON.parse(stored);
            if (storageItem.expiration && Date.now() > storageItem.expiration) {
              expiredItems++;
            }
          }
        } catch (error) {
          // Count corrupted items as expired
          expiredItems++;
        }
      });
    } catch (error) {
      console.error('Failed to get storage stats:', error);
    }

    return { totalItems, totalSize, expiredItems };
  }

  /**
   * Check if storage is available and functional
   */
  isAvailable(): boolean {
    if (!browser) return false;

    try {
      const testKey = `${this.prefix}__test__`;
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Export all data for backup
   */
  exportData(): Record<string, any> {
    if (!browser) return {};

    const data: Record<string, any> = {};
    try {
      const keys = Object.keys(localStorage).filter(k => k.startsWith(this.prefix));
      
      keys.forEach(fullKey => {
        try {
          const stored = localStorage.getItem(fullKey);
          if (stored) {
            const shortKey = fullKey.replace(this.prefix, '');
            data[shortKey] = JSON.parse(stored);
          }
        } catch (error) {
          console.error('Failed to export item:', fullKey, error);
        }
      });
    } catch (error) {
      console.error('Failed to export data:', error);
    }
    return data;
  }

  /**
   * Import data from backup
   */
  importData(data: Record<string, any>, overwrite: boolean = false): number {
    if (!browser) return 0;

    let imported = 0;
    try {
      Object.entries(data).forEach(([key, storageItem]) => {
        const fullKey = `${this.prefix}${key}`;
        
        if (!overwrite && localStorage.getItem(fullKey)) {
          return; // Skip existing items
        }

        try {
          localStorage.setItem(fullKey, JSON.stringify(storageItem));
          imported++;
        } catch (error) {
          console.error('Failed to import item:', key, error);
        }
      });
    } catch (error) {
      console.error('Failed to import data:', error);
    }
    return imported;
  }
}

// Create default instance
export const sessionStorage = new SessionStorageManager();

// Research Agent specific storage keys
export const STORAGE_KEYS = {
  ACTIVE_SESSION: 'active_session',
  RESEARCH_SESSIONS: 'research_sessions',
  USER_PREFERENCES: 'user_preferences',
  CHAT_SESSIONS: 'chat_sessions',
  REPORT_CONFIGS: 'report_configs',
  CACHE_INDEX: 'cache_index',
  ANALYTICS_DATA: 'analytics_data'
} as const;

// Type-safe storage helpers for specific data types
export class ResearchSessionStorage {
  private storage = sessionStorage;

  storeActiveSession(session: any): boolean {
    return this.storage.setItem(STORAGE_KEYS.ACTIVE_SESSION, session, 8 * 60 * 60 * 1000); // 8 hours
  }

  getActiveSession(): any | null {
    return this.storage.getItem(STORAGE_KEYS.ACTIVE_SESSION);
  }

  clearActiveSession(): boolean {
    return this.storage.removeItem(STORAGE_KEYS.ACTIVE_SESSION);
  }

  storeResearchSessions(sessions: any[]): boolean {
    return this.storage.setItem(STORAGE_KEYS.RESEARCH_SESSIONS, sessions, 7 * 24 * 60 * 60 * 1000); // 7 days
  }

  getResearchSessions(defaultValue: any[] = []): any[] {
    return this.storage.getItem(STORAGE_KEYS.RESEARCH_SESSIONS, defaultValue);
  }

  addResearchSession(session: any): boolean {
    const sessions = this.getResearchSessions();
    const updated = [session, ...sessions.filter(s => s.id !== session.id)].slice(0, 50); // Keep max 50
    return this.storeResearchSessions(updated);
  }

  storeUserPreferences(preferences: any): boolean {
    return this.storage.setItem(STORAGE_KEYS.USER_PREFERENCES, preferences, 30 * 24 * 60 * 60 * 1000); // 30 days
  }

  getUserPreferences(defaultValue: any = {}): any {
    return this.storage.getItem(STORAGE_KEYS.USER_PREFERENCES, defaultValue);
  }

  storeChatSession(chatSessionId: string, messages: any[]): boolean {
    const chatSessions = this.storage.getItem(STORAGE_KEYS.CHAT_SESSIONS, {});
    chatSessions[chatSessionId] = {
      messages,
      lastUpdated: Date.now(),
      messageCount: messages.length
    };
    return this.storage.setItem(STORAGE_KEYS.CHAT_SESSIONS, chatSessions, 24 * 60 * 60 * 1000); // 24 hours
  }

  getChatSession(chatSessionId: string): any[] {
    const chatSessions = this.storage.getItem(STORAGE_KEYS.CHAT_SESSIONS, {});
    return chatSessions[chatSessionId]?.messages || [];
  }

  storeReportConfig(configName: string, config: any): boolean {
    const configs = this.storage.getItem(STORAGE_KEYS.REPORT_CONFIGS, {});
    configs[configName] = {
      config,
      savedAt: Date.now()
    };
    return this.storage.setItem(STORAGE_KEYS.REPORT_CONFIGS, configs);
  }

  getReportConfigs(): Record<string, any> {
    return this.storage.getItem(STORAGE_KEYS.REPORT_CONFIGS, {});
  }

  // Analytics data caching
  storeAnalyticsData(sessionId: string, data: any): boolean {
    const analyticsData = this.storage.getItem(STORAGE_KEYS.ANALYTICS_DATA, {});
    analyticsData[sessionId] = {
      data,
      generatedAt: Date.now()
    };
    return this.storage.setItem(STORAGE_KEYS.ANALYTICS_DATA, analyticsData, 2 * 60 * 60 * 1000); // 2 hours
  }

  getAnalyticsData(sessionId: string): any | null {
    const analyticsData = this.storage.getItem(STORAGE_KEYS.ANALYTICS_DATA, {});
    return analyticsData[sessionId]?.data || null;
  }

  // Cache management
  updateCacheIndex(): boolean {
    const stats = this.storage.getStats();
    const index = {
      lastUpdated: Date.now(),
      stats,
      sessions: Object.keys(this.storage.getItem(STORAGE_KEYS.RESEARCH_SESSIONS, [])).length,
      chatSessions: Object.keys(this.storage.getItem(STORAGE_KEYS.CHAT_SESSIONS, {})).length
    };
    return this.storage.setItem(STORAGE_KEYS.CACHE_INDEX, index);
  }

  getCacheStats(): any {
    return this.storage.getItem(STORAGE_KEYS.CACHE_INDEX, {});
  }

  // Cleanup utilities
  performMaintenance(): { cleared: number; errors: number } {
    let cleared = 0;
    let errors = 0;

    try {
      // Clear expired items
      cleared += this.storage.clearExpired();

      // Clean up old chat sessions (keep only last 10)
      const chatSessions = this.storage.getItem(STORAGE_KEYS.CHAT_SESSIONS, {});
      const sortedSessions = Object.entries(chatSessions)
        .sort(([,a], [,b]) => (b as any).lastUpdated - (a as any).lastUpdated)
        .slice(0, 10);
      
      if (sortedSessions.length !== Object.keys(chatSessions).length) {
        const cleaned = Object.fromEntries(sortedSessions);
        this.storage.setItem(STORAGE_KEYS.CHAT_SESSIONS, cleaned);
        cleared += Object.keys(chatSessions).length - sortedSessions.length;
      }

      // Update cache index
      this.updateCacheIndex();
      
    } catch (error) {
      console.error('Maintenance error:', error);
      errors++;
    }

    return { cleared, errors };
  }
}

// Create default research session storage instance
export const researchSessionStorage = new ResearchSessionStorage();