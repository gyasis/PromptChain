/**
 * Comprehensive error handling system for Research Agent frontend
 * Handles API errors, validation errors, and user-facing error messages
 */

import { researchSessionStorage } from '../stores/sessionStorage';

export enum ErrorType {
  NETWORK = 'network',
  VALIDATION = 'validation', 
  API = 'api',
  SESSION = 'session',
  FILE_SYSTEM = 'file_system',
  WEBSOCKET = 'websocket',
  UNKNOWN = 'unknown'
}

export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export interface ResearchError {
  id: string;
  type: ErrorType;
  severity: ErrorSeverity;
  message: string;
  details?: any;
  timestamp: Date;
  sessionId?: string;
  userId?: string;
  stack?: string;
  context?: Record<string, any>;
}

export interface ErrorHandlerOptions {
  showToast?: boolean;
  logToConsole?: boolean;
  sendToServer?: boolean;
  persistLocal?: boolean;
}

class ErrorHandlerService {
  private errorQueue: ResearchError[] = [];
  private maxQueueSize = 100;
  private isOnline = navigator.onLine;

  constructor() {
    // Monitor online status
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.flushErrorQueue();
    });
    
    window.addEventListener('offline', () => {
      this.isOnline = false;
    });

    // Global error handlers
    window.addEventListener('error', (event) => {
      this.handleError({
        type: ErrorType.UNKNOWN,
        severity: ErrorSeverity.HIGH,
        message: event.message,
        details: {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno
        },
        stack: event.error?.stack
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.handleError({
        type: ErrorType.UNKNOWN,
        severity: ErrorSeverity.HIGH,
        message: 'Unhandled Promise Rejection',
        details: event.reason,
        stack: event.reason?.stack
      });
    });
  }

  /**
   * Main error handling method
   */
  handleError(
    error: Partial<ResearchError>,
    options: ErrorHandlerOptions = {}
  ): string {
    const errorId = this.generateErrorId();
    
    const fullError: ResearchError = {
      id: errorId,
      type: error.type || ErrorType.UNKNOWN,
      severity: error.severity || ErrorSeverity.MEDIUM,
      message: error.message || 'An unexpected error occurred',
      details: error.details,
      timestamp: new Date(),
      sessionId: error.sessionId,
      userId: error.userId,
      stack: error.stack,
      context: error.context || this.getErrorContext()
    };

    // Default options
    const opts = {
      showToast: true,
      logToConsole: true,
      sendToServer: false,
      persistLocal: true,
      ...options
    };

    // Process error based on options
    if (opts.logToConsole) {
      this.logToConsole(fullError);
    }

    if (opts.persistLocal) {
      this.persistError(fullError);
    }

    if (opts.sendToServer && this.isOnline) {
      this.sendErrorToServer(fullError);
    } else if (opts.sendToServer) {
      this.queueError(fullError);
    }

    if (opts.showToast) {
      this.showUserNotification(fullError);
    }

    return errorId;
  }

  /**
   * Handle API errors with specific context
   */
  handleApiError(
    response: Response,
    context?: Record<string, any>,
    options?: ErrorHandlerOptions
  ): string {
    const errorType = this.getApiErrorType(response.status);
    const severity = this.getApiErrorSeverity(response.status);
    
    return this.handleError({
      type: errorType,
      severity,
      message: this.getApiErrorMessage(response.status),
      details: {
        status: response.status,
        statusText: response.statusText,
        url: response.url
      },
      context: {
        ...context,
        apiCall: true,
        endpoint: response.url
      }
    }, options);
  }

  /**
   * Handle validation errors
   */
  handleValidationError(
    field: string,
    value: any,
    rule: string,
    context?: Record<string, any>
  ): string {
    return this.handleError({
      type: ErrorType.VALIDATION,
      severity: ErrorSeverity.LOW,
      message: `Validation failed for ${field}: ${rule}`,
      details: {
        field,
        value,
        rule
      },
      context
    });
  }

  /**
   * Handle session-related errors
   */
  handleSessionError(
    sessionId: string,
    operation: string,
    details?: any
  ): string {
    return this.handleError({
      type: ErrorType.SESSION,
      severity: ErrorSeverity.MEDIUM,
      message: `Session error during ${operation}`,
      sessionId,
      details
    });
  }

  /**
   * Handle WebSocket errors
   */
  handleWebSocketError(
    sessionId: string,
    error: Event | Error,
    context?: Record<string, any>
  ): string {
    return this.handleError({
      type: ErrorType.WEBSOCKET,
      severity: ErrorSeverity.HIGH,
      message: 'WebSocket connection error',
      sessionId,
      details: error,
      context: {
        ...context,
        connectionType: 'websocket'
      }
    });
  }

  /**
   * Get recent errors for debugging
   */
  getRecentErrors(limit = 10): ResearchError[] {
    const errors = researchSessionStorage.storage.getItem('error_log', []);
    return errors.slice(-limit);
  }

  /**
   * Clear error log
   */
  clearErrorLog(): void {
    researchSessionStorage.storage.removeItem('error_log');
    this.errorQueue = [];
  }

  /**
   * Get error statistics
   */
  getErrorStats(): {
    total: number;
    byType: Record<ErrorType, number>;
    bySeverity: Record<ErrorSeverity, number>;
    recent24h: number;
  } {
    const errors = researchSessionStorage.storage.getItem('error_log', []);
    const now = new Date();
    const oneDayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);

    const byType = Object.values(ErrorType).reduce((acc, type) => ({
      ...acc,
      [type]: 0
    }), {} as Record<ErrorType, number>);

    const bySeverity = Object.values(ErrorSeverity).reduce((acc, severity) => ({
      ...acc,
      [severity]: 0
    }), {} as Record<ErrorSeverity, number>);

    let recent24h = 0;

    errors.forEach((error: ResearchError) => {
      byType[error.type]++;
      bySeverity[error.severity]++;
      
      if (new Date(error.timestamp) > oneDayAgo) {
        recent24h++;
      }
    });

    return {
      total: errors.length,
      byType,
      bySeverity,
      recent24h
    };
  }

  // Private methods
  private generateErrorId(): string {
    return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private getErrorContext(): Record<string, any> {
    return {
      url: window.location.href,
      userAgent: navigator.userAgent,
      timestamp: new Date().toISOString(),
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      online: navigator.onLine
    };
  }

  private logToConsole(error: ResearchError): void {
    const prefix = `[${error.severity.toUpperCase()}] ${error.type}`;
    
    switch (error.severity) {
      case ErrorSeverity.CRITICAL:
        console.error(prefix, error);
        break;
      case ErrorSeverity.HIGH:
        console.error(prefix, error);
        break;
      case ErrorSeverity.MEDIUM:
        console.warn(prefix, error);
        break;
      case ErrorSeverity.LOW:
        console.info(prefix, error);
        break;
    }
  }

  private persistError(error: ResearchError): void {
    try {
      const errors = researchSessionStorage.storage.getItem('error_log', []);
      errors.push(error);
      
      // Keep only last 1000 errors
      const trimmedErrors = errors.slice(-1000);
      researchSessionStorage.storage.setItem('error_log', trimmedErrors);
    } catch (e) {
      console.warn('Failed to persist error to local storage:', e);
    }
  }

  private queueError(error: ResearchError): void {
    this.errorQueue.push(error);
    
    // Prevent memory issues
    if (this.errorQueue.length > this.maxQueueSize) {
      this.errorQueue = this.errorQueue.slice(-this.maxQueueSize);
    }
  }

  private async sendErrorToServer(error: ResearchError): Promise<void> {
    try {
      await fetch('/api/errors', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(error)
      });
    } catch (e) {
      console.warn('Failed to send error to server:', e);
      this.queueError(error);
    }
  }

  private flushErrorQueue(): void {
    if (this.errorQueue.length === 0) return;

    const errorsToSend = [...this.errorQueue];
    this.errorQueue = [];

    errorsToSend.forEach(error => {
      this.sendErrorToServer(error);
    });
  }

  private getApiErrorType(status: number): ErrorType {
    if (status >= 500) return ErrorType.API;
    if (status >= 400) return ErrorType.VALIDATION;
    return ErrorType.NETWORK;
  }

  private getApiErrorSeverity(status: number): ErrorSeverity {
    if (status >= 500) return ErrorSeverity.HIGH;
    if (status === 401 || status === 403) return ErrorSeverity.HIGH;
    if (status >= 400) return ErrorSeverity.MEDIUM;
    return ErrorSeverity.LOW;
  }

  private getApiErrorMessage(status: number): string {
    const messages: Record<number, string> = {
      400: 'Bad Request - Please check your input',
      401: 'Unauthorized - Please log in again',
      403: 'Forbidden - You do not have permission',
      404: 'Not Found - The requested resource was not found',
      408: 'Request Timeout - Please try again',
      429: 'Too Many Requests - Please wait before trying again',
      500: 'Internal Server Error - Please contact support',
      502: 'Bad Gateway - Service temporarily unavailable',
      503: 'Service Unavailable - Please try again later',
      504: 'Gateway Timeout - Please try again'
    };

    return messages[status] || `HTTP Error ${status}`;
  }

  private showUserNotification(error: ResearchError): void {
    // This would integrate with a toast notification system
    // For now, we'll use console notification
    console.info('User notification:', error.message);
    
    // In production, this would show a user-friendly toast/modal
    // Example: showToast(error.message, error.severity);
  }
}

// Create singleton instance
export const errorHandler = new ErrorHandlerService();

// Utility functions for common error scenarios
export const handleApiCall = async <T>(
  apiCall: () => Promise<Response>,
  context?: Record<string, any>
): Promise<T> => {
  try {
    const response = await apiCall();
    
    if (!response.ok) {
      errorHandler.handleApiError(response, context);
      throw new Error(`API call failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    if (error instanceof Error) {
      errorHandler.handleError({
        type: ErrorType.NETWORK,
        severity: ErrorSeverity.HIGH,
        message: error.message,
        stack: error.stack,
        context
      });
    }
    throw error;
  }
};

export const withErrorBoundary = <T extends any[], R>(
  fn: (...args: T) => R,
  context?: Record<string, any>
) => {
  return (...args: T): R => {
    try {
      return fn(...args);
    } catch (error) {
      if (error instanceof Error) {
        errorHandler.handleError({
          type: ErrorType.UNKNOWN,
          severity: ErrorSeverity.HIGH,
          message: error.message,
          stack: error.stack,
          context: {
            ...context,
            functionName: fn.name,
            arguments: args
          }
        });
      }
      throw error;
    }
  };
};