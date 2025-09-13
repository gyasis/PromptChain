"""
HTTP client for making API requests with authentication support
"""
import os
import requests
import time
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.utils.logger import get_logger

logger = get_logger(__name__)

class APIClient:
    """REST API client with authentication and retry support."""

    def __init__(self):
        self.base_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
        self.timeout = int(os.getenv('API_TIMEOUT', 30))
        self.session = requests.Session()
        self.auth_token = None
        self.api_key = os.getenv('API_KEY')
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=int(os.getenv('RETRY_COUNT', 3)),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Setup authentication
        self._setup_authentication()

    def _setup_authentication(self):
        """Setup authentication based on environment variables."""
        if self.api_key:
            self.session.headers['X-API-Key'] = self.api_key
            logger.info("Using API Key authentication")
        elif os.getenv('ADMIN_EMAIL') and os.getenv('ADMIN_PASSWORD'):
            self._authenticate_with_credentials()

    def _authenticate_with_credentials(self):
        """Authenticate using email/password to get JWT token."""
        try:
            auth_data = {
                'email': os.getenv('ADMIN_EMAIL'),
                'password': os.getenv('ADMIN_PASSWORD')
            }
            response = self.post('/auth/login', auth_data)
            if response.status_code == 200:
                self.auth_token = response.json().get('token')
                self.session.headers['Authorization'] = f'Bearer {self.auth_token}'
                logger.info("Successfully authenticated with JWT token")
            else:
                logger.warning("Failed to authenticate with credentials")
        except Exception as e:
            logger.warning(f"Authentication failed: {e}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with logging and error handling."""
        url = f"{self.base_url.rstrip('/')}{endpoint}"
        
        # Log request
        logger.info(f"→ {method.upper()} {endpoint}")
        if kwargs.get('json'):
            logger.debug(f"Request data: {kwargs['json']}")
        
        start_time = time.time()
        try:
            response = self.session.request(method, url, timeout=self.timeout, **kwargs)
            duration = time.time() - start_time
            
            # Log response
            logger.info(f"← {response.status_code} {method.upper()} {endpoint} ({duration:.2f}s)")
            
            if response.status_code >= 400:
                logger.error(f"Error response: {response.text}")
            
            return response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get(self, endpoint: str, params: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make GET request."""
        return self._make_request('GET', endpoint, params=params, **kwargs)

    def post(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make POST request."""
        return self._make_request('POST', endpoint, json=data, **kwargs)

    def put(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make PUT request."""
        return self._make_request('PUT', endpoint, json=data, **kwargs)

    def patch(self, endpoint: str, data: Optional[Dict] = None, **kwargs) -> requests.Response:
        """Make PATCH request."""
        return self._make_request('PATCH', endpoint, json=data, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> requests.Response:
        """Make DELETE request."""
        return self._make_request('DELETE', endpoint, **kwargs)

    def upload_file(self, endpoint: str, file_path: str, field_name: str = 'file') -> requests.Response:
        """Upload file via multipart form data."""
        with open(file_path, 'rb') as f:
            files = {field_name: f}
            # Remove Content-Type header for file uploads
            headers = dict(self.session.headers)
            headers.pop('Content-Type', None)
            return self._make_request('POST', endpoint, files=files, headers=headers)



