"""
Configuration state synchronization system for Research Agent.

Provides real-time synchronization of configuration changes between CLI and web processes
using WebSocket notifications, file watching, and shared state management.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigChangeType(Enum):
    """Types of configuration changes"""
    MODEL_ADDED = "model_added"
    MODEL_UPDATED = "model_updated"  
    MODEL_REMOVED = "model_removed"
    PROFILE_CHANGED = "profile_changed"
    TASK_ASSIGNMENT_CHANGED = "task_assignment_changed"
    SETTINGS_UPDATED = "settings_updated"
    FULL_RELOAD = "full_reload"


@dataclass
class ConfigChangeEvent:
    """Configuration change event"""
    id: str
    timestamp: float
    change_type: ConfigChangeType
    source_process: str  # CLI, Web, System
    affected_keys: List[str]
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "change_type": self.change_type.value,
            "source_process": self.source_process,
            "affected_keys": self.affected_keys,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigChangeEvent":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            change_type=ConfigChangeType(data["change_type"]),
            source_process=data["source_process"],
            affected_keys=data["affected_keys"],
            old_value=data.get("old_value"),
            new_value=data.get("new_value"),
            metadata=data.get("metadata", {})
        )


class ConfigSyncSubscriber:
    """Interface for configuration change subscribers"""
    
    async def on_config_changed(self, event: ConfigChangeEvent) -> None:
        """Called when configuration changes"""
        pass


class WebSocketConfigSyncSubscriber(ConfigSyncSubscriber):
    """WebSocket-based configuration sync subscriber for web interface"""
    
    def __init__(self, websocket_manager=None):
        self.websocket_manager = websocket_manager
        self.connected_clients: Set[Any] = set()
    
    def add_client(self, websocket):
        """Add WebSocket client for notifications"""
        self.connected_clients.add(websocket)
    
    def remove_client(self, websocket):
        """Remove WebSocket client"""
        self.connected_clients.discard(websocket)
    
    async def on_config_changed(self, event: ConfigChangeEvent) -> None:
        """Notify all connected web clients of configuration change"""
        if not self.connected_clients:
            return
        
        message = {
            "type": "config_changed",
            "event": event.to_dict()
        }
        
        # Send to all connected clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(message))
            except Exception as e:
                logger.warning(f"Failed to send config update to WebSocket client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.connected_clients.discard(client)


class ConfigurationSyncManager:
    """
    Manages configuration state synchronization between processes.
    
    Provides real-time notifications when configuration changes occur,
    allowing CLI and web processes to stay synchronized.
    """
    
    def __init__(self, process_id: Optional[str] = None):
        self.process_id = process_id or f"process_{uuid.uuid4().hex[:8]}"
        self.subscribers: List[ConfigSyncSubscriber] = []
        self._lock = threading.RLock()
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task = None
        self._event_history: List[ConfigChangeEvent] = []
        self._max_history = 100
        
    def add_subscriber(self, subscriber: ConfigSyncSubscriber):
        """Add configuration change subscriber"""
        with self._lock:
            self.subscribers.append(subscriber)
    
    def remove_subscriber(self, subscriber: ConfigSyncSubscriber):
        """Remove configuration change subscriber"""
        with self._lock:
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)
    
    def notify_change(self, change_type: ConfigChangeType, affected_keys: List[str],
                     old_value: Optional[Any] = None, new_value: Optional[Any] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Notify subscribers of configuration change"""
        event = ConfigChangeEvent(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            change_type=change_type,
            source_process=self.process_id,
            affected_keys=affected_keys,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {}
        )
        
        # Add to event queue for async processing
        try:
            self._event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Config sync event queue full, dropping oldest event")
            try:
                self._event_queue.get_nowait()
                self._event_queue.put_nowait(event)
            except asyncio.QueueEmpty:
                pass
        
        # Add to history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        
        logger.debug(f"Config change notified: {change_type.value} affecting {affected_keys}")
    
    async def start(self):
        """Start the synchronization manager"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info(f"Configuration sync manager started (process: {self.process_id})")
    
    async def stop(self):
        """Stop the synchronization manager"""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Configuration sync manager stopped")
    
    async def _process_events(self):
        """Process configuration change events"""
        while self._running:
            try:
                # Wait for events with timeout
                try:
                    event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Notify all subscribers
                with self._lock:
                    subscribers = self.subscribers.copy()
                
                for subscriber in subscribers:
                    try:
                        await subscriber.on_config_changed(event)
                    except Exception as e:
                        logger.error(f"Error notifying subscriber {type(subscriber).__name__}: {e}")
                
                # Mark task done
                self._event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing config sync event: {e}")
    
    def get_event_history(self, limit: Optional[int] = None) -> List[ConfigChangeEvent]:
        """Get recent configuration change history"""
        with self._lock:
            if limit:
                return self._event_history[-limit:]
            return self._event_history.copy()
    
    def get_process_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        with self._lock:
            return {
                "process_id": self.process_id,
                "running": self._running,
                "subscribers": len(self.subscribers),
                "event_history_size": len(self._event_history),
                "queue_size": self._event_queue.qsize()
            }


class ModelConfigSyncAdapter:
    """
    Adapter to integrate ModelConfigManager with ConfigurationSyncManager.
    
    Monitors model configuration changes and triggers sync notifications.
    """
    
    def __init__(self, model_config_manager, sync_manager: ConfigurationSyncManager):
        self.model_config_manager = model_config_manager
        self.sync_manager = sync_manager
        self._previous_config = {}
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup hooks to detect configuration changes"""
        # Add callback to model config manager
        self.model_config_manager.add_reload_callback(self._on_config_reloaded)
        
        # Store initial configuration
        self._previous_config = self.model_config_manager.get_config_data()
    
    def _on_config_reloaded(self):
        """Handle configuration reload"""
        current_config = self.model_config_manager.get_config_data()
        self._detect_and_notify_changes(self._previous_config, current_config)
        self._previous_config = current_config
    
    def _detect_and_notify_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Detect and notify specific configuration changes"""
        try:
            # Check for model changes
            old_models = old_config.get("models", {})
            new_models = new_config.get("models", {})
            
            # Detect added models
            for model_key in new_models:
                if model_key not in old_models:
                    self.sync_manager.notify_change(
                        ConfigChangeType.MODEL_ADDED,
                        [f"models.{model_key}"],
                        old_value=None,
                        new_value=new_models[model_key]
                    )
            
            # Detect removed models
            for model_key in old_models:
                if model_key not in new_models:
                    self.sync_manager.notify_change(
                        ConfigChangeType.MODEL_REMOVED,
                        [f"models.{model_key}"],
                        old_value=old_models[model_key],
                        new_value=None
                    )
            
            # Detect updated models
            for model_key in new_models:
                if model_key in old_models and old_models[model_key] != new_models[model_key]:
                    self.sync_manager.notify_change(
                        ConfigChangeType.MODEL_UPDATED,
                        [f"models.{model_key}"],
                        old_value=old_models[model_key],
                        new_value=new_models[model_key]
                    )
            
            # Check for profile changes
            old_profile = old_config.get("active_profile")
            new_profile = new_config.get("active_profile")
            if old_profile != new_profile:
                self.sync_manager.notify_change(
                    ConfigChangeType.PROFILE_CHANGED,
                    ["active_profile"],
                    old_value=old_profile,
                    new_value=new_profile
                )
            
            # Check for task assignment changes
            old_tasks = old_config.get("task_specific_models", {})
            new_tasks = new_config.get("task_specific_models", {})
            if old_tasks != new_tasks:
                self.sync_manager.notify_change(
                    ConfigChangeType.TASK_ASSIGNMENT_CHANGED,
                    ["task_specific_models"],
                    old_value=old_tasks,
                    new_value=new_tasks
                )
            
            # Check for settings changes
            old_settings = old_config.get("litellm_settings", {})
            new_settings = new_config.get("litellm_settings", {})
            if old_settings != new_settings:
                self.sync_manager.notify_change(
                    ConfigChangeType.SETTINGS_UPDATED,
                    ["litellm_settings"],
                    old_value=old_settings,
                    new_value=new_settings
                )
        
        except Exception as e:
            logger.error(f"Error detecting configuration changes: {e}")
            # Fallback: notify full reload
            self.sync_manager.notify_change(
                ConfigChangeType.FULL_RELOAD,
                ["*"],
                metadata={"error": str(e)}
            )


# Global synchronization manager
_global_sync_manager: Optional[ConfigurationSyncManager] = None
_sync_lock = threading.Lock()


def get_config_sync_manager(process_id: Optional[str] = None) -> ConfigurationSyncManager:
    """Get or create global configuration synchronization manager"""
    global _global_sync_manager
    
    with _sync_lock:
        if _global_sync_manager is None:
            _global_sync_manager = ConfigurationSyncManager(process_id)
        return _global_sync_manager


def setup_model_config_sync(model_config_manager, process_id: Optional[str] = None) -> ConfigurationSyncManager:
    """
    Setup synchronization for model configuration manager.
    
    Args:
        model_config_manager: ModelConfigManager instance
        process_id: Optional process identifier
        
    Returns:
        ConfigurationSyncManager instance
    """
    sync_manager = get_config_sync_manager(process_id)
    adapter = ModelConfigSyncAdapter(model_config_manager, sync_manager)
    
    logger.info(f"Model configuration synchronization setup complete (process: {sync_manager.process_id})")
    return sync_manager


async def demo_sync_system():
    """Demo the configuration sync system"""
    # Create sync manager
    sync_manager = ConfigurationSyncManager("demo_process")
    
    # Create WebSocket subscriber (simulated)
    class DemoSubscriber(ConfigSyncSubscriber):
        async def on_config_changed(self, event: ConfigChangeEvent):
            print(f"[DEMO] Config changed: {event.change_type.value} -> {event.affected_keys}")
    
    subscriber = DemoSubscriber()
    sync_manager.add_subscriber(subscriber)
    
    # Start sync manager
    await sync_manager.start()
    
    # Simulate configuration changes
    sync_manager.notify_change(
        ConfigChangeType.MODEL_ADDED,
        ["models.new_model"],
        new_value={"model": "gpt-4o", "temperature": 0.7}
    )
    
    sync_manager.notify_change(
        ConfigChangeType.PROFILE_CHANGED,
        ["active_profile"],
        old_value="balanced",
        new_value="premium"
    )
    
    # Wait a bit for processing
    await asyncio.sleep(0.1)
    
    # Show stats
    stats = sync_manager.get_process_stats()
    print(f"[DEMO] Sync stats: {stats}")
    
    # Stop sync manager
    await sync_manager.stop()


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_sync_system())