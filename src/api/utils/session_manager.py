"""
Session Manager

Manages user sessions, agent state, and conversation history.
"""

import asyncio
import uuid
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import logging

from ...agents import DataAnalyst

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Session data structure."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    user_id: Optional[str] = None
    
    # Agent state
    data_analyst: Optional[DataAnalyst] = None
    current_dataset_id: Optional[str] = None
    
    # Conversation history
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Session metadata
    total_queries: int = 0
    total_charts: int = 0
    total_datasets: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session data to dictionary."""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'user_id': self.user_id,
            'current_dataset_id': self.current_dataset_id,
            'conversation_history': self.conversation_history,
            'total_queries': self.total_queries,
            'total_charts': self.total_charts,
            'total_datasets': self.total_datasets
        }


class SessionManager:
    """
    Manages user sessions and agent state.
    """
    
    def __init__(self, session_timeout: int = 3600, cleanup_interval: int = 300):
        """
        Initialize session manager.
        
        Args:
            session_timeout: Session timeout in seconds (default: 1 hour)
            cleanup_interval: Cleanup interval in seconds (default: 5 minutes)
        """
        self.session_timeout = session_timeout
        self.cleanup_interval = cleanup_interval
        self.sessions: Dict[str, SessionData] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        
        # Session storage
        self.session_storage_path = Path("sessions")
        self.session_storage_path.mkdir(exist_ok=True)
        
        logger.info(f"SessionManager initialized with timeout={session_timeout}s")
    
    async def initialize(self):
        """Initialize the session manager."""
        logger.info("Starting session manager...")
        
        # Load existing sessions
        await self._load_sessions()
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info("Session manager started successfully")
    
    async def cleanup(self):
        """Cleanup session manager."""
        logger.info("Stopping session manager...")
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Save all sessions
        await self._save_all_sessions()
        
        logger.info("Session manager stopped")
    
    async def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        now = datetime.now()
        
        # Create session data
        session_data = SessionData(
            session_id=session_id,
            created_at=now,
            last_activity=now,
            user_id=user_id
        )
        
        # Initialize DataAnalyst for this session
        session_data.data_analyst = DataAnalyst()
        
        # Store session
        self.sessions[session_id] = session_data
        
        # Save session to disk
        await self._save_session(session_id)
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data or None if not found
        """
        session_data = self.sessions.get(session_id)
        
        if session_data:
            # Update last activity
            session_data.last_activity = datetime.now()
            
            # Check if session is expired
            if self._is_session_expired(session_data):
                await self.delete_session(session_id)
                return None
            
            return session_data
        
        return None
    
    async def get_or_create_session(self, session_id: Optional[str] = None, user_id: Optional[str] = None) -> str:
        """
        Get existing session or create new one.
        
        Args:
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Returns:
            Session ID
        """
        if session_id:
            session_data = await self.get_session(session_id)
            if session_data:
                return session_id
        
        # Create new session
        return await self.create_session(user_id)
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            
            # Remove session file
            session_file = self.session_storage_path / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()
            
            logger.info(f"Deleted session: {session_id}")
            return True
        
        return False
    
    async def get_data_analyst(self, session_id: str) -> Optional[DataAnalyst]:
        """
        Get DataAnalyst for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            DataAnalyst instance or None
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            if not session_data.data_analyst:
                # Initialize if not exists
                session_data.data_analyst = DataAnalyst()
            return session_data.data_analyst
        
        return None
    
    async def add_conversation_entry(self, session_id: str, entry: Dict[str, Any]):
        """
        Add entry to conversation history.
        
        Args:
            session_id: Session identifier
            entry: Conversation entry
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            # Add timestamp
            entry['timestamp'] = datetime.now().isoformat()
            
            # Add to history
            session_data.conversation_history.append(entry)
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            # Update counters
            if entry.get('type') == 'query':
                session_data.total_queries += 1
            elif entry.get('type') == 'chart':
                session_data.total_charts += 1
            elif entry.get('type') == 'dataset':
                session_data.total_datasets += 1
            
            # Save session
            await self._save_session(session_id)
    
    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of entries to return
            
        Returns:
            List of conversation entries
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            return session_data.conversation_history[-limit:]
        
        return []
    
    async def set_current_dataset(self, session_id: str, dataset_id: str):
        """
        Set current dataset for a session.
        
        Args:
            session_id: Session identifier
            dataset_id: Dataset identifier
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            session_data.current_dataset_id = dataset_id
            await self._save_session(session_id)
    
    async def get_current_dataset(self, session_id: str) -> Optional[str]:
        """
        Get current dataset for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dataset identifier or None
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            return session_data.current_dataset_id
        
        return None
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session statistics
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            now = datetime.now()
            session_duration = (now - session_data.created_at).total_seconds()
            
            return {
                'session_id': session_id,
                'created_at': session_data.created_at.isoformat(),
                'duration_seconds': session_duration,
                'total_queries': session_data.total_queries,
                'total_charts': session_data.total_charts,
                'total_datasets': session_data.total_datasets,
                'conversation_entries': len(session_data.conversation_history),
                'current_dataset_id': session_data.current_dataset_id
            }
        
        return {}
    
    async def store_dataset_metadata(self, session_id: str, dataset_id: str, metadata: Dict[str, Any]):
        """
        Store dataset metadata in session.
        
        Args:
            session_id: Session identifier
            dataset_id: Dataset identifier
            metadata: Dataset metadata to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            # Set current dataset
            session_data.current_dataset_id = dataset_id
            
            # Add to conversation history
            entry = {
                'type': 'dataset',
                'action': 'upload',
                'dataset_id': dataset_id,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            session_data.total_datasets += 1
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            # Save session
            await self._save_session(session_id)
    
    async def store_analysis_results(self, session_id: str, analysis_id: str, results: Dict[str, Any]):
        """
        Store analysis results in session.
        
        Args:
            session_id: Session identifier
            analysis_id: Analysis identifier
            results: Analysis results to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            entry = {
                'type': 'analysis',
                'action': 'completed',
                'analysis_id': analysis_id,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            session_data.total_queries += 1
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            await self._save_session(session_id)
    
    async def store_query_results(self, session_id: str, query_id: str, results: Dict[str, Any]):
        """
        Store query results in session.
        
        Args:
            session_id: Session identifier
            query_id: Query identifier
            results: Query results to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            entry = {
                'type': 'query',
                'action': 'completed',
                'query_id': query_id,
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            session_data.total_queries += 1
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            await self._save_session(session_id)
    
    async def store_report(self, session_id: str, report_id: str, report_data: Dict[str, Any]):
        """
        Store report in session.
        
        Args:
            session_id: Session identifier
            report_id: Report identifier
            report_data: Report data to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            entry = {
                'type': 'report',
                'action': 'generated',
                'report_id': report_id,
                'report_data': report_data,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            await self._save_session(session_id)
    
    async def store_dashboard(self, session_id: str, dashboard_id: str, dashboard_data: Dict[str, Any]):
        """
        Store dashboard in session.
        
        Args:
            session_id: Session identifier
            dashboard_id: Dashboard identifier
            dashboard_data: Dashboard data to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            entry = {
                'type': 'dashboard',
                'action': 'created',
                'dashboard_id': dashboard_id,
                'dashboard_data': dashboard_data,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            session_data.total_charts += 1
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            await self._save_session(session_id)
    
    async def store_insights(self, session_id: str, insight_id: str, insights_data: Dict[str, Any]):
        """
        Store insights in session.
        
        Args:
            session_id: Session identifier
            insight_id: Insight identifier
            insights_data: Insights data to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            entry = {
                'type': 'insights',
                'action': 'generated',
                'insight_id': insight_id,
                'insights_data': insights_data,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            await self._save_session(session_id)
    
    async def store_visualization(self, session_id: str, viz_id: str, visualization_data: Dict[str, Any]):
        """
        Store visualization in session.
        
        Args:
            session_id: Session identifier
            viz_id: Visualization identifier
            visualization_data: Visualization data to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            entry = {
                'type': 'visualization',
                'action': 'created',
                'viz_id': viz_id,
                'visualization_data': visualization_data,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            session_data.total_charts += 1
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            await self._save_session(session_id)
    
    async def store_chart(self, session_id: str, chart_id: str, chart_data: Dict[str, Any]):
        """
        Store chart in session.
        
        Args:
            session_id: Session identifier
            chart_id: Chart identifier
            chart_data: Chart data to store
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            entry = {
                'type': 'chart',
                'action': 'created',
                'chart_id': chart_id,
                'chart_data': chart_data,
                'timestamp': datetime.now().isoformat()
            }
            
            session_data.conversation_history.append(entry)
            session_data.total_charts += 1
            
            # Limit history size
            max_history = 100
            if len(session_data.conversation_history) > max_history:
                session_data.conversation_history = session_data.conversation_history[-max_history:]
            
            await self._save_session(session_id)
    
    async def get_analysis_results(self, session_id: str, analysis_id: str) -> Optional[Dict[str, Any]]:
        """
        Get analysis results from session.
        
        Args:
            session_id: Session identifier
            analysis_id: Analysis identifier
            
        Returns:
            Analysis results or None if not found
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            for entry in reversed(session_data.conversation_history):
                if (entry.get('type') == 'analysis' and 
                    entry.get('analysis_id') == analysis_id):
                    return entry.get('results')
        
        return None
    
    async def get_chart(self, session_id: str, chart_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chart data from session.
        
        Args:
            session_id: Session identifier
            chart_id: Chart identifier
            
        Returns:
            Chart data or None if not found
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            for entry in reversed(session_data.conversation_history):
                if (entry.get('type') == 'chart' and 
                    entry.get('chart_id') == chart_id):
                    return entry.get('chart_data')
        
        return None
    
    async def get_dashboard(self, session_id: str, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """
        Get dashboard data from session.
        
        Args:
            session_id: Session identifier
            dashboard_id: Dashboard identifier
            
        Returns:
            Dashboard data or None if not found
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            for entry in reversed(session_data.conversation_history):
                if (entry.get('type') == 'dashboard' and 
                    entry.get('dashboard_id') == dashboard_id):
                    return entry.get('dashboard_data')
        
        return None
    
    async def remove_dataset_from_session(self, session_id: str, dataset_id: str):
        """
        Remove dataset from session.
        
        Args:
            session_id: Session identifier
            dataset_id: Dataset identifier to remove
        """
        session_data = await self.get_session(session_id)
        
        if session_data:
            # Remove dataset entries from conversation history
            session_data.conversation_history = [
                entry for entry in session_data.conversation_history
                if not (entry.get('type') == 'dataset' and 
                       entry.get('dataset_id') == dataset_id)
            ]
            
            # Reset current dataset if it was the removed one
            if session_data.current_dataset_id == dataset_id:
                session_data.current_dataset_id = None
            
            await self._save_session(session_id)
    
    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all sessions.
        
        Args:
            user_id: Optional user identifier to filter by
            
        Returns:
            List of session information
        """
        sessions = []
        
        for session_data in self.sessions.values():
            if user_id and session_data.user_id != user_id:
                continue
            
            sessions.append({
                'session_id': session_data.session_id,
                'created_at': session_data.created_at.isoformat(),
                'last_activity': session_data.last_activity.isoformat(),
                'user_id': session_data.user_id,
                'total_queries': session_data.total_queries,
                'total_charts': session_data.total_charts,
                'total_datasets': session_data.total_datasets
            })
        
        return sessions
    
    def _is_session_expired(self, session_data: SessionData) -> bool:
        """Check if session is expired."""
        now = datetime.now()
        time_since_activity = (now - session_data.last_activity).total_seconds()
        return time_since_activity > self.session_timeout
    
    async def _cleanup_expired_sessions(self):
        """Cleanup expired sessions periodically."""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                expired_sessions = []
                for session_id, session_data in self.sessions.items():
                    if self._is_session_expired(session_data):
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    await self.delete_session(session_id)
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error during session cleanup: {e}")
    
    async def _save_session(self, session_id: str):
        """Save session to disk."""
        session_data = self.sessions.get(session_id)
        if not session_data:
            return
        
        try:
            session_file = self.session_storage_path / f"{session_id}.json"
            with open(session_file, 'w') as f:
                # Save session data (without agent instance)
                data = session_data.to_dict()
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
    
    async def _save_all_sessions(self):
        """Save all sessions to disk."""
        for session_id in self.sessions:
            await self._save_session(session_id)
    
    async def _load_sessions(self):
        """Load sessions from disk."""
        try:
            for session_file in self.session_storage_path.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        data = json.load(f)
                    
                    # Create session data
                    session_data = SessionData(
                        session_id=data['session_id'],
                        created_at=datetime.fromisoformat(data['created_at']),
                        last_activity=datetime.fromisoformat(data['last_activity']),
                        user_id=data.get('user_id'),
                        current_dataset_id=data.get('current_dataset_id'),
                        conversation_history=data.get('conversation_history', []),
                        total_queries=data.get('total_queries', 0),
                        total_charts=data.get('total_charts', 0),
                        total_datasets=data.get('total_datasets', 0)
                    )
                    
                    # Initialize DataAnalyst
                    session_data.data_analyst = DataAnalyst()
                    
                    # Check if session is expired
                    if not self._is_session_expired(session_data):
                        self.sessions[session_data.session_id] = session_data
                    else:
                        # Remove expired session file
                        session_file.unlink()
                        
                except Exception as e:
                    logger.error(f"Error loading session from {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global session statistics."""
        total_sessions = len(self.sessions)
        total_queries = sum(s.total_queries for s in self.sessions.values())
        total_charts = sum(s.total_charts for s in self.sessions.values())
        total_datasets = sum(s.total_datasets for s in self.sessions.values())
        
        return {
            'total_sessions': total_sessions,
            'total_queries': total_queries,
            'total_charts': total_charts,
            'total_datasets': total_datasets,
            'session_timeout': self.session_timeout,
            'cleanup_interval': self.cleanup_interval
        } 