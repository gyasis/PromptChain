/**
 * Comprehensive logging system for Research Agent frontend
 * Supports structured logging, performance tracking, and user analytics
 */

import { researchSessionStorage } from '../stores/sessionStorage';
import { errorHandler, ErrorSeverity, ErrorType } from './errorHandler';

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
  CRITICAL = 4
}

export enum LogCategory {
  SYSTEM = 'system',
  USER = 'user',
  API = 'api',
  PERFORMANCE = 'performance',
  RESEARCH = 'research',
  SESSION = 'session',
  WEBSOCKET = 'websocket',
  UI = 'ui'
}

export interface LogEntry {
  id: string;
  timestamp: Date;
  level: LogLevel;
  category: LogCategory;
  message: string;
  data?: any;
  sessionId?: string;
  userId?: string;
  context?: Record<string, any>;
  performance?: {
    duration?: number;
    memory?: number;
    cpu?: number;
  };
}

export interface LoggerConfig {
  level: LogLevel;
  enableConsole: boolean;
  enableStorage: boolean;
  enableServer: boolean;
  maxStorageEntries: number;
  flushInterval: number;
  categories: LogCategory[];
}

class LoggerService {
  private config: LoggerConfig;
  private logBuffer: LogEntry[] = [];
  private performanceMarks: Map<string, number> = new Map();
  private sessionId: string;

  constructor(config: Partial<LoggerConfig> = {}) {
    this.config = {
      level: LogLevel.INFO,
      enableConsole: true,
      enableStorage: true,
      enableServer: false,
      maxStorageEntries: 1000,
      flushInterval: 30000, // 30 seconds
      categories: Object.values(LogCategory),
      ...config
    };

    this.sessionId = this.generateSessionId();
    this.initializeLogging();
  }

  /**
   * Debug level logging
   */
  debug(
    message: string,
    data?: any,
    category: LogCategory = LogCategory.SYSTEM,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.DEBUG, category, message, data, context);
  }

  /**
   * Info level logging
   */
  info(
    message: string,
    data?: any,
    category: LogCategory = LogCategory.SYSTEM,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.INFO, category, message, data, context);
  }

  /**
   * Warning level logging
   */
  warn(
    message: string,
    data?: any,
    category: LogCategory = LogCategory.SYSTEM,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.WARN, category, message, data, context);
  }

  /**
   * Error level logging
   */
  error(
    message: string,
    error?: Error | any,
    category: LogCategory = LogCategory.SYSTEM,
    context?: Record<string, any>
  ): void {
    const logData = error instanceof Error ? {
      name: error.name,
      message: error.message,
      stack: error.stack,
      ...error
    } : error;

    this.log(LogLevel.ERROR, category, message, logData, context);

    // Also send to error handler for comprehensive error tracking
    errorHandler.handleError({
      type: ErrorType.UNKNOWN,
      severity: ErrorSeverity.HIGH,
      message,
      details: logData,
      context
    }, { showToast: false });
  }

  /**
   * Critical level logging
   */
  critical(
    message: string,
    data?: any,
    category: LogCategory = LogCategory.SYSTEM,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.CRITICAL, category, message, data, context);

    // Also send to error handler
    errorHandler.handleError({
      type: ErrorType.UNKNOWN,
      severity: ErrorSeverity.CRITICAL,
      message,
      details: data,
      context
    });
  }

  /**
   * Log research-specific events
   */
  logResearchEvent(
    event: string,
    sessionId: string,
    data?: any,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.INFO, LogCategory.RESEARCH, `Research: ${event}`, data, {
      ...context,
      researchSessionId: sessionId
    });
  }

  /**
   * Log API calls with performance tracking
   */
  logApiCall(
    method: string,
    url: string,
    duration: number,
    status: number,
    data?: any
  ): void {
    const level = status >= 400 ? LogLevel.ERROR : LogLevel.INFO;
    
    this.log(level, LogCategory.API, `API ${method} ${url}`, {
      method,
      url,
      status,
      response: data
    }, undefined, {
      duration
    });
  }

  /**
   * Log user interactions
   */
  logUserAction(
    action: string,
    element?: string,
    data?: any,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.INFO, LogCategory.USER, `User: ${action}`, {
      element,
      ...data
    }, context);
  }

  /**
   * Log UI events and state changes
   */
  logUIEvent(
    event: string,
    component?: string,
    data?: any,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.DEBUG, LogCategory.UI, `UI: ${event}`, {
      component,
      ...data
    }, context);
  }

  /**
   * Start performance tracking
   */
  startPerformanceTracking(operation: string): void {
    this.performanceMarks.set(operation, performance.now());
    this.debug(`Performance tracking started: ${operation}`, undefined, LogCategory.PERFORMANCE);
  }

  /**
   * End performance tracking and log results
   */
  endPerformanceTracking(
    operation: string,
    data?: any,
    context?: Record<string, any>
  ): number {
    const startTime = this.performanceMarks.get(operation);
    if (!startTime) {
      this.warn(`Performance tracking not found for: ${operation}`, undefined, LogCategory.PERFORMANCE);
      return 0;
    }

    const duration = performance.now() - startTime;
    this.performanceMarks.delete(operation);

    this.log(LogLevel.INFO, LogCategory.PERFORMANCE, `Performance: ${operation}`, data, context, {
      duration,
      memory: (performance as any).memory?.usedJSHeapSize,
      cpu: this.getCPUUsage()
    });

    return duration;
  }

  /**
   * Log session events
   */
  logSessionEvent(
    event: string,
    sessionId: string,
    data?: any,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.INFO, LogCategory.SESSION, `Session: ${event}`, data, {
      ...context,
      targetSessionId: sessionId
    });
  }

  /**
   * Log WebSocket events
   */
  logWebSocketEvent(
    event: string,
    sessionId: string,
    data?: any,
    context?: Record<string, any>
  ): void {
    this.log(LogLevel.INFO, LogCategory.WEBSOCKET, `WebSocket: ${event}`, data, {
      ...context,
      connectionSessionId: sessionId
    });
  }

  /**
   * Get recent logs for debugging
   */
  getRecentLogs(limit = 100): LogEntry[] {
    const logs = researchSessionStorage.storage.getItem('app_logs', []);
    return logs.slice(-limit);
  }

  /**
   * Get logs by category
   */
  getLogsByCategory(category: LogCategory, limit = 50): LogEntry[] {
    const logs = researchSessionStorage.storage.getItem('app_logs', []);
    return logs
      .filter((log: LogEntry) => log.category === category)
      .slice(-limit);
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): {
    averageApiDuration: number;
    totalRequests: number;
    errorRate: number;
    memoryUsage: number;
  } {
    const logs = this.getLogsByCategory(LogCategory.PERFORMANCE);
    const apiLogs = this.getLogsByCategory(LogCategory.API);

    const apiDurations = logs
      .filter(log => log.performance?.duration)
      .map(log => log.performance!.duration!);

    const errorCount = apiLogs.filter(log => log.level >= LogLevel.ERROR).length;

    return {
      averageApiDuration: apiDurations.length > 0 
        ? apiDurations.reduce((a, b) => a + b, 0) / apiDurations.length 
        : 0,
      totalRequests: apiLogs.length,
      errorRate: apiLogs.length > 0 ? errorCount / apiLogs.length : 0,
      memoryUsage: (performance as any).memory?.usedJSHeapSize || 0
    };
  }

  /**
   * Clear all logs
   */
  clearLogs(): void {
    researchSessionStorage.storage.removeItem('app_logs');
    this.logBuffer = [];
    this.info('Logs cleared', undefined, LogCategory.SYSTEM);
  }

  /**
   * Export logs for debugging
   */
  exportLogs(): string {
    const logs = researchSessionStorage.storage.getItem('app_logs', []);
    return JSON.stringify(logs, null, 2);
  }

  /**
   * Update logger configuration
   */
  updateConfig(newConfig: Partial<LoggerConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.info('Logger configuration updated', newConfig, LogCategory.SYSTEM);
  }

  // Private methods
  private log(
    level: LogLevel,
    category: LogCategory,
    message: string,
    data?: any,
    context?: Record<string, any>,
    performance?: LogEntry['performance']
  ): void {
    // Check if logging is enabled for this level and category
    if (level < this.config.level || !this.config.categories.includes(category)) {
      return;
    }

    const logEntry: LogEntry = {
      id: this.generateLogId(),
      timestamp: new Date(),
      level,
      category,
      message,
      data,
      sessionId: this.sessionId,
      context: {
        url: window.location.href,
        userAgent: navigator.userAgent,
        ...context
      },
      performance
    };

    // Output to console if enabled
    if (this.config.enableConsole) {
      this.logToConsole(logEntry);
    }

    // Add to buffer for storage/server
    this.logBuffer.push(logEntry);

    // Store locally if enabled
    if (this.config.enableStorage) {
      this.persistLog(logEntry);
    }

    // Flush buffer periodically
    if (this.logBuffer.length >= 50) {
      this.flushBuffer();
    }
  }

  private initializeLogging(): void {
    // Set up periodic buffer flushing
    setInterval(() => {
      this.flushBuffer();
    }, this.config.flushInterval);

    // Log system initialization
    this.info('Logger initialized', {
      config: this.config,
      sessionId: this.sessionId
    }, LogCategory.SYSTEM);

    // Set up page visibility change logging
    document.addEventListener('visibilitychange', () => {
      this.logUserAction('page_visibility_change', 'document', {
        hidden: document.hidden
      });
    });

    // Log page load completion
    if (document.readyState === 'complete') {
      this.logPerformanceEvent('page_load_complete');
    } else {
      window.addEventListener('load', () => {
        this.logPerformanceEvent('page_load_complete');
      });
    }
  }

  private logToConsole(entry: LogEntry): void {
    const prefix = `[${LogLevel[entry.level]}] ${entry.category}`;
    const timestamp = entry.timestamp.toISOString();
    
    const logMessage = `${timestamp} ${prefix}: ${entry.message}`;
    
    switch (entry.level) {
      case LogLevel.DEBUG:
        console.debug(logMessage, entry.data);
        break;
      case LogLevel.INFO:
        console.info(logMessage, entry.data);
        break;
      case LogLevel.WARN:
        console.warn(logMessage, entry.data);
        break;
      case LogLevel.ERROR:
      case LogLevel.CRITICAL:
        console.error(logMessage, entry.data);
        break;
    }
  }

  private persistLog(entry: LogEntry): void {
    try {
      const logs = researchSessionStorage.storage.getItem('app_logs', []);
      logs.push(entry);
      
      // Keep only the most recent entries
      const trimmedLogs = logs.slice(-this.config.maxStorageEntries);
      researchSessionStorage.storage.setItem('app_logs', trimmedLogs);
    } catch (error) {
      console.warn('Failed to persist log entry:', error);
    }
  }

  private flushBuffer(): void {
    if (this.logBuffer.length === 0) return;

    if (this.config.enableServer && navigator.onLine) {
      this.sendLogsToServer([...this.logBuffer]);
    }

    this.logBuffer = [];
  }

  private async sendLogsToServer(logs: LogEntry[]): Promise<void> {
    try {
      await fetch('/api/logs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ logs })
      });
    } catch (error) {
      console.warn('Failed to send logs to server:', error);
    }
  }

  private generateLogId(): string {
    return `log_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getCPUUsage(): number {
    // Basic CPU usage estimation (not accurate, but helpful for trends)
    return 0; // Would need Web Workers for proper CPU monitoring
  }

  private logPerformanceEvent(event: string): void {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    
    this.log(LogLevel.INFO, LogCategory.PERFORMANCE, `Performance: ${event}`, {
      loadTime: navigation.loadEventEnd - navigation.loadEventStart,
      domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
      pageSize: document.documentElement.innerHTML.length
    });
  }
}

// Create singleton instance
export const logger = new LoggerService();

// Utility functions for common logging patterns
export const logApiCall = (method: string, url: string) => {
  const startTime = performance.now();
  
  return {
    success: (status: number, data?: any) => {
      const duration = performance.now() - startTime;
      logger.logApiCall(method, url, duration, status, data);
    },
    error: (status: number, error?: any) => {
      const duration = performance.now() - startTime;
      logger.logApiCall(method, url, duration, status, error);
    }
  };
};

export const withLogging = <T extends any[], R>(
  fn: (...args: T) => R,
  operationName: string,
  category: LogCategory = LogCategory.SYSTEM
) => {
  return (...args: T): R => {
    logger.startPerformanceTracking(operationName);
    
    try {
      const result = fn(...args);
      logger.endPerformanceTracking(operationName, { success: true }, {
        functionName: fn.name,
        arguments: args
      });
      return result;
    } catch (error) {
      logger.endPerformanceTracking(operationName, { success: false, error }, {
        functionName: fn.name,
        arguments: args
      });
      throw error;
    }
  };
};