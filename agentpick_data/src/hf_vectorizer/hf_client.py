"""HuggingFace API client for downloading models and metadata."""

import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from .utils import retry
from .config import HF_TOKEN, MIN_DOWNLOADS_THRESHOLD

logger = logging.getLogger(__name__)


class HFClient:
    """Client for HuggingFace API operations."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize HF API client.
        
        Args:
            token: HuggingFace API token (defaults to env var HF_TOKEN)
        """
        self.token = token or HF_TOKEN
        self.hf_api = HfApi(token=self.token)
    
    def fetch_all_models(self, limit: Optional[int] = None) -> List[str]:
        """Fetch all text-generation model IDs from HuggingFace Hub.
        
        Uses API-side sorting by downloads (descending) with early break once
        downloads < 1000 threshold is reached. This drastically reduces dataset
        size (~80-95% reduction) by avoiding low-popularity models at the API level.
        
        Args:
            limit: Maximum number of models to fetch (None for all)
            
        Returns:
            List of model IDs (pre-filtered to downloads >= 1000 where possible)
        """
        logger.info("Fetching text-generation models sorted by downloads (descending)...")
        
        def fetch_models():
            models = []
            kwargs = {
                'filter': 'text-generation',
                'sort': 'downloads',
                'direction': -1,
            }
            if limit:
                kwargs['limit'] = limit
            
            for model in self.hf_api.list_models(**kwargs):
                # Early break: stop once we hit models with < 1000 downloads
                if hasattr(model, 'downloads') and model.downloads is not None:
                    if model.downloads < MIN_DOWNLOADS_THRESHOLD:
                        logger.info(f"Early break at {model.id} with {model.downloads} downloads")
                        break
                
                models.append(model.id)
                if limit and len(models) >= limit:
                    break
            
            return models
        
        try:
            models = retry(fetch_models, retries=3, delay=2)
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            raise
        
        logger.info(f"Found {len(models)} qualified text-generation models")
        return models
    
    def download_readme(self, model_id: str, model_dir: Path) -> Optional[Path]:
        """Download README.md for a model with retries.
        
        Args:
            model_id: Model ID (e.g., 'gpt2')
            model_dir: Directory to download to
            
        Returns:
            Path to downloaded README or None if not found
        """
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            def download():
                return hf_hub_download(
                    repo_id=model_id,
                    filename="README.md",
                    repo_type="model",
                    local_dir=model_dir,
                    force_download=False,
                    token=self.token,
                )
            
            readme_path = retry(download, retries=3, delay=2)
            return Path(readme_path)
            
        except FileNotFoundError:
            logger.debug(f"No README.md found for {model_id}")
            return None
        except Exception as e:
            if "Repository not found" in str(e) or "not found" in str(e).lower():
                logger.warning(f"Repository not found: {model_id}")
            else:
                logger.warning(f"Error downloading README for {model_id}: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch model metadata from HuggingFace Hub with retries.
        
        Distinguishes between API failure and real 0 downloads to avoid silent
        skipping of valid models due to transient errors.
        
        Args:
            model_id: Model ID
            
        Returns:
            Dictionary with model metadata, or None if fetch fails
        """
        try:
            def fetch():
                return self.hf_api.model_info(repo_id=model_id)
            
            model_info = retry(fetch, retries=3, delay=2)
            return {
                'downloads': model_info.downloads,
                'likes': model_info.likes,
                'tags': model_info.tags or [],
                'pipeline_tag': model_info.pipeline_tag,
                'library_name': model_info.library_name,
                'created_at': model_info.created_at.isoformat() if model_info.created_at else None,
                'last_modified': model_info.last_modified.isoformat() if model_info.last_modified else None,
            }
        except Exception as e:
            logger.warning(f"Error fetching model info for {model_id}: {e}")
            return None
