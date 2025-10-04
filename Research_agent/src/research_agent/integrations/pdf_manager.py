"""
PDF Download and Storage Manager

Handles downloading, organizing, and managing PDFs from various sources.
"""

import os
import json
import asyncio
import hashlib
import logging
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from urllib.parse import urlparse, unquote
import sqlite3
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


@dataclass
class PDFMetadata:
    """Metadata for PDF documents"""
    title: str
    url: str
    local_path: str
    file_hash: str
    download_date: datetime
    file_size: int
    source: str
    authors: Optional[List[str]] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    publication_date: Optional[datetime] = None


class PDFManager:
    """
    Manages PDF downloads, storage, and metadata tracking
    """
    
    def __init__(self, base_path: str = "./papers", config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF Manager
        
        Args:
            base_path: Base directory for storing PDFs
            config: Optional configuration dict
        """
        self.base_path = Path(base_path)
        self.config = config or {}
        
        # Create base directory structure
        self._setup_storage_structure()
        
        # Initialize metadata database
        self.db_path = self.base_path / "pdf_metadata.db"
        self._init_database()
        
        # Configuration
        self.max_retries = self.config.get('max_retries', 3)
        self.timeout = self.config.get('timeout', 60)
        self.chunk_size = self.config.get('chunk_size', 8192)
        self.max_concurrent_downloads = self.config.get('max_concurrent_downloads', 5)
        
        # Download statistics
        self.download_stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0,
            'already_exists': 0,
            'total_size_bytes': 0
        }
    
    def _setup_storage_structure(self):
        """Create organized folder structure for PDF storage"""
        # Main structure: papers/source/year/filename.pdf
        sources = ['arxiv', 'pubmed', 'sci_hub', 'other']
        current_year = datetime.now().year
        years = range(current_year - 5, current_year + 1)  # Last 5 years + current
        
        for source in sources:
            source_dir = self.base_path / source
            source_dir.mkdir(parents=True, exist_ok=True)
            
            # Create year subdirectories
            for year in years:
                year_dir = source_dir / str(year)
                year_dir.mkdir(exist_ok=True)
        
        # Create metadata and cache directories
        (self.base_path / 'metadata').mkdir(exist_ok=True)
        (self.base_path / 'cache').mkdir(exist_ok=True)
        
        logger.info(f"PDF storage structure initialized at: {self.base_path}")
    
    def _init_database(self):
        """Initialize SQLite database for PDF metadata tracking"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Create PDF metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pdf_metadata (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                source TEXT NOT NULL,
                year INTEGER,
                doi TEXT,
                arxiv_id TEXT,
                pmid TEXT,
                file_path TEXT NOT NULL,
                file_hash TEXT UNIQUE,
                file_size INTEGER,
                download_url TEXT,
                download_date TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON pdf_metadata(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON pdf_metadata(year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doi ON pdf_metadata(doi)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON pdf_metadata(file_hash)")
        
        # Create download history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS download_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pdf_id TEXT,
                url TEXT,
                status TEXT,
                file_size INTEGER,
                download_time_ms INTEGER,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (pdf_id) REFERENCES pdf_metadata(id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("PDF metadata database initialized")
    
    async def download_pdf(
        self, 
        url: str, 
        paper_metadata: Dict[str, Any],
        force_download: bool = False
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Download a PDF from URL and store it in organized structure
        
        Args:
            url: PDF download URL
            paper_metadata: Metadata about the paper
            force_download: Force re-download even if file exists
            
        Returns:
            Tuple of (success, file_path, download_info)
        """
        self.download_stats['total_attempted'] += 1
        
        try:
            # Generate storage path
            storage_path = self._generate_storage_path(paper_metadata)
            
            # Check if already downloaded (unless forced)
            if not force_download and storage_path.exists():
                file_hash = self._calculate_file_hash(storage_path)
                existing_record = self._get_metadata_by_hash(file_hash)
                
                if existing_record:
                    self.download_stats['already_exists'] += 1
                    logger.info(f"PDF already exists: {storage_path}")
                    return True, str(storage_path), {
                        'status': 'already_exists',
                        'file_size': storage_path.stat().st_size,
                        'file_hash': file_hash
                    }
            
            # Download with retry logic
            download_success = False
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    logger.info(f"Downloading PDF (attempt {attempt + 1}): {url}")
                    
                    start_time = datetime.now()
                    
                    # Use aiohttp for async download
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, timeout=self.timeout) as response:
                            response.raise_for_status()
                            
                            # Verify content type
                            content_type = response.headers.get('content-type', '').lower()
                            if 'pdf' not in content_type and not await self._is_pdf_content(response):
                                raise ValueError(f"Content is not a PDF: {content_type}")
                            
                            # Download to temporary file first
                            temp_path = storage_path.with_suffix('.tmp')
                            temp_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            with open(temp_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(self.chunk_size):
                                    f.write(chunk)
                            
                            # Verify downloaded file
                            if not self._verify_pdf_file(temp_path):
                                raise ValueError("Downloaded file is not a valid PDF")
                            
                            # Move to final location
                            temp_path.rename(storage_path)
                            
                            download_time = (datetime.now() - start_time).total_seconds() * 1000
                            file_size = storage_path.stat().st_size
                            file_hash = self._calculate_file_hash(storage_path)
                            
                            # Store metadata
                            self._store_metadata(paper_metadata, storage_path, url, file_hash, file_size)
                            
                            # Record download history
                            self._record_download(
                                paper_metadata.get('id'),
                                url,
                                'success',
                                file_size,
                                download_time
                            )
                            
                            self.download_stats['successful'] += 1
                            self.download_stats['total_size_bytes'] += file_size
                            
                            logger.info(f"Successfully downloaded: {storage_path} ({file_size} bytes)")
                            
                            return True, str(storage_path), {
                                'status': 'downloaded',
                                'file_size': file_size,
                                'file_hash': file_hash,
                                'download_time_ms': download_time,
                                'attempts': attempt + 1
                            }
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # All attempts failed
            self.download_stats['failed'] += 1
            self._record_download(
                paper_metadata.get('id'),
                url,
                'failed',
                0,
                0,
                str(last_error)
            )
            
            logger.error(f"Failed to download PDF after {self.max_retries} attempts: {url}")
            return False, "", {
                'status': 'failed',
                'error': str(last_error),
                'attempts': self.max_retries
            }
            
        except Exception as e:
            logger.error(f"PDF download error: {e}")
            self.download_stats['failed'] += 1
            return False, "", {
                'status': 'error',
                'error': str(e)
            }
    
    async def _is_pdf_content(self, response) -> bool:
        """Check if response content is a PDF by examining first bytes"""
        try:
            # Read first 5 bytes to check PDF header
            first_bytes = await response.content.read(5)
            return first_bytes.startswith(b'%PDF-')
        except:
            return False
    
    def _generate_storage_path(self, paper_metadata: Dict[str, Any]) -> Path:
        """Generate organized storage path for PDF"""
        source = paper_metadata.get('source', 'other').lower()
        year = paper_metadata.get('publication_year', datetime.now().year)
        
        # Generate filename
        paper_id = paper_metadata.get('id', '')
        title = paper_metadata.get('title', 'untitled')
        
        # Clean title for filename
        clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_'))[:50]
        clean_title = clean_title.replace(' ', '_')
        
        # Create unique filename
        if paper_id:
            filename = f"{paper_id}_{clean_title}.pdf"
        else:
            # Use hash of title + authors for uniqueness
            unique_str = f"{title}_{paper_metadata.get('authors', [])}"
            unique_hash = hashlib.md5(unique_str.encode()).hexdigest()[:8]
            filename = f"{clean_title}_{unique_hash}.pdf"
        
        # Build path: papers/source/year/filename.pdf
        storage_path = self.base_path / source / str(year) / filename
        
        return storage_path
    
    def _verify_pdf_file(self, file_path: Path) -> bool:
        """Verify if a file is a valid PDF"""
        try:
            if not file_path.exists():
                return False
            
            if file_path.stat().st_size < 100:  # Too small to be valid PDF
                return False
            
            # Check PDF header
            with open(file_path, 'rb') as f:
                header = f.read(5)
                return header.startswith(b'%PDF-')
                
        except Exception:
            return False
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _store_metadata(
        self, 
        paper_metadata: Dict[str, Any],
        file_path: Path,
        download_url: str,
        file_hash: str,
        file_size: int
    ):
        """Store PDF metadata in database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO pdf_metadata (
                    id, title, authors, source, year, doi, arxiv_id, pmid,
                    file_path, file_hash, file_size, download_url, download_date,
                    metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                paper_metadata.get('id'),
                paper_metadata.get('title'),
                json.dumps(paper_metadata.get('authors', [])),
                paper_metadata.get('source'),
                paper_metadata.get('publication_year'),
                paper_metadata.get('doi'),
                paper_metadata.get('metadata', {}).get('arxiv_id'),
                paper_metadata.get('metadata', {}).get('pmid'),
                str(file_path),
                file_hash,
                file_size,
                download_url,
                datetime.now().isoformat(),
                json.dumps(paper_metadata.get('metadata', {}))
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def _record_download(
        self,
        pdf_id: str,
        url: str,
        status: str,
        file_size: int,
        download_time_ms: float,
        error_message: str = None
    ):
        """Record download attempt in history"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO download_history (
                    pdf_id, url, status, file_size, download_time_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (pdf_id, url, status, file_size, download_time_ms, error_message))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def _get_metadata_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get PDF metadata by file hash"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM pdf_metadata WHERE file_hash = ?
            """, (file_hash,))
            
            row = cursor.fetchone()
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            
            return None
            
        finally:
            conn.close()
    
    async def download_papers_batch(
        self,
        papers: List[Dict[str, Any]],
        force_download: bool = False
    ) -> Dict[str, Any]:
        """
        Download multiple papers in parallel
        
        Args:
            papers: List of paper metadata dicts with download URLs
            force_download: Force re-download even if files exist
            
        Returns:
            Dictionary with download results
        """
        logger.info(f"Starting batch download of {len(papers)} papers")
        
        results = {
            'successful': [],
            'failed': [],
            'already_exists': [],
            'total_size_bytes': 0
        }
        
        # Create download tasks
        tasks = []
        for paper in papers:
            url = paper.get('pdf_url') or paper.get('url')
            if url:
                task = self.download_pdf(url, paper, force_download)
                tasks.append((paper.get('id'), task))
        
        # Execute downloads with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
        
        async def download_with_limit(paper_id, task):
            async with semaphore:
                success, file_path, info = await task
                return paper_id, success, file_path, info
        
        # Execute all downloads
        download_results = await asyncio.gather(
            *[download_with_limit(pid, task) for pid, task in tasks],
            return_exceptions=True
        )
        
        # Process results
        for result in download_results:
            if isinstance(result, Exception):
                logger.error(f"Download exception: {result}")
                results['failed'].append({'error': str(result)})
            else:
                paper_id, success, file_path, info = result
                
                if success:
                    if info['status'] == 'already_exists':
                        results['already_exists'].append({
                            'paper_id': paper_id,
                            'file_path': file_path,
                            'info': info
                        })
                    else:
                        results['successful'].append({
                            'paper_id': paper_id,
                            'file_path': file_path,
                            'info': info
                        })
                        results['total_size_bytes'] += info.get('file_size', 0)
                else:
                    results['failed'].append({
                        'paper_id': paper_id,
                        'info': info
                    })
        
        # Log summary
        logger.info(f"Batch download complete: {len(results['successful'])} successful, "
                   f"{len(results['already_exists'])} already existed, "
                   f"{len(results['failed'])} failed")
        
        return results
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get statistics about PDF storage"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Get counts by source
            cursor.execute("""
                SELECT source, COUNT(*) as count, SUM(file_size) as total_size
                FROM pdf_metadata
                GROUP BY source
            """)
            source_stats = {row[0]: {'count': row[1], 'size': row[2]} 
                          for row in cursor.fetchall()}
            
            # Get counts by year
            cursor.execute("""
                SELECT year, COUNT(*) as count
                FROM pdf_metadata
                GROUP BY year
                ORDER BY year DESC
            """)
            year_stats = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get total statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_pdfs,
                    SUM(file_size) as total_size,
                    AVG(file_size) as avg_size,
                    MAX(file_size) as max_size,
                    MIN(file_size) as min_size
                FROM pdf_metadata
            """)
            totals = cursor.fetchone()
            
            # Get recent downloads
            cursor.execute("""
                SELECT COUNT(*) as recent_count
                FROM pdf_metadata
                WHERE download_date > datetime('now', '-7 days')
            """)
            recent_count = cursor.fetchone()[0]
            
            return {
                'total_pdfs': totals[0] or 0,
                'total_size_bytes': totals[1] or 0,
                'total_size_mb': round((totals[1] or 0) / (1024 * 1024), 2),
                'average_size_bytes': totals[2] or 0,
                'max_size_bytes': totals[3] or 0,
                'min_size_bytes': totals[4] or 0,
                'by_source': source_stats,
                'by_year': year_stats,
                'recent_downloads': recent_count,
                'storage_path': str(self.base_path),
                'download_stats': self.download_stats
            }
            
        finally:
            conn.close()
    
    def search_pdfs(
        self,
        query: Optional[str] = None,
        source: Optional[str] = None,
        year: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for PDFs in the database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            sql = "SELECT * FROM pdf_metadata WHERE 1=1"
            params = []
            
            if query:
                sql += " AND (title LIKE ? OR authors LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])
            
            if source:
                sql += " AND source = ?"
                params.append(source)
            
            if year:
                sql += " AND year = ?"
                params.append(year)
            
            sql += " ORDER BY download_date DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        finally:
            conn.close()
    
    def cleanup_duplicates(self) -> int:
        """Remove duplicate PDFs based on file hash"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Find duplicates
            cursor.execute("""
                SELECT file_hash, COUNT(*) as count
                FROM pdf_metadata
                GROUP BY file_hash
                HAVING count > 1
            """)
            
            duplicates_removed = 0
            
            for file_hash, count in cursor.fetchall():
                # Keep the oldest, remove newer duplicates
                cursor.execute("""
                    SELECT id, file_path
                    FROM pdf_metadata
                    WHERE file_hash = ?
                    ORDER BY created_at
                """, (file_hash,))
                
                entries = cursor.fetchall()
                
                # Remove all but the first
                for entry in entries[1:]:
                    pdf_id, file_path = entry
                    
                    # Delete file
                    try:
                        Path(file_path).unlink()
                    except:
                        pass
                    
                    # Delete metadata
                    cursor.execute("DELETE FROM pdf_metadata WHERE id = ?", (pdf_id,))
                    duplicates_removed += 1
            
            conn.commit()
            
            logger.info(f"Removed {duplicates_removed} duplicate PDFs")
            return duplicates_removed
            
        finally:
            conn.close()