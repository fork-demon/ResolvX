"""
SharePoint integration tools for file operations.

These tools provide direct SharePoint integration for:
- Listing files in SharePoint folders (CSV uploads from business users)
- Downloading files from SharePoint (CSV validation)
- Searching SharePoint documents (verify CSV in archive)

IMPORTANT: These tools should ONLY be used for:
- Competitor promotional file processing failures
- Basket segment file processing failures
- Verifying if CSV moved from process/ to archive/ folder
"""

from typing import Any, Dict, Optional
from pathlib import Path
import tempfile
import os

from core.observability import get_logger, get_tracer
from core.gateway.tool_registry import BaseTool


class SharePointListFilesTool(BaseTool):
    """
    Tool to list CSV files in SharePoint folders.
    
    ONLY use for file processing failures to check:
    - process/ folder: CSV files waiting to be processed
    - archive/ folder: CSV files that have been successfully processed
    """
    
    def __init__(self, site_url: Optional[str] = None, **config):
        """Initialize SharePoint list files tool."""
        self.name = "sharepoint_list_files"
        self.description = "List CSV files in SharePoint process/ or archive/ folder - ONLY for file processing failures"
        
        # Load from environment variables (configured in .env)
        self.site_url = site_url or os.getenv("SHAREPOINT_SITE_URL", "https://tesco.sharepoint.com/sites/pricing")
        self.process_folder = os.getenv("SHAREPOINT_PROCESS_FOLDER", "/Shared Documents/CSV_Uploads/Process")
        self.archive_folder = os.getenv("SHAREPOINT_ARCHIVE_FOLDER", "/Shared Documents/CSV_Uploads/Archive")
        
        self.mock_mode = True  # Enable mock mode for local dev
        self.logger = get_logger("tools.sharepoint.list")
        self.tracer = get_tracer("tools.sharepoint.list")
        
        self.logger.info(f"SharePoint configured: process={self.process_folder}, archive={self.archive_folder}")
    
    def validate_parameters(self, **parameters: Any) -> bool:
        """Validate tool parameters."""
        return "folder_path" in parameters
    
    async def execute(self, **parameters: Any) -> Dict[str, Any]:
        """Execute the list files operation."""
        folder_path = parameters.get("folder_path", "Shared Documents")
        recursive = parameters.get("recursive", False)
        file_extension = parameters.get("file_extension")
        
        with self.tracer.start_as_current_span("sharepoint_list_files_exec") as span:
            # Set input using our custom wrapper's method
            if hasattr(span, 'set_input'):
                span.set_input({"folder_path": folder_path, "recursive": recursive, "file_extension": file_extension})
            
            # Mock implementation
            files = [
                {
                    "name": "basket_segments_troubleshooting.pdf",
                    "path": f"{folder_path}/basket_segments_troubleshooting.pdf",
                    "size": 245678,
                    "modified": "2025-10-20T14:30:00Z",
                    "type": "file",
                    "extension": "pdf"
                },
                {
                    "name": "price_lifecycle_runbook.docx",
                    "path": f"{folder_path}/price_lifecycle_runbook.docx",
                    "size": 123456,
                    "modified": "2025-10-19T10:15:00Z",
                    "type": "file",
                    "extension": "docx"
                }
            ]
            
            if file_extension:
                files = [f for f in files if f["extension"] == file_extension.lower()]
            
            result = {
                "success": True,
                "folder_path": folder_path,
                "files": files,
                "count": len(files),
                "mock": True
            }
            
            span.set_output(result)
            self.logger.info(f"Listed {len(files)} files from {folder_path}")
            return result


class SharePointDownloadFileTool(BaseTool):
    """Tool to download a file from SharePoint."""
    
    def __init__(self, site_url: Optional[str] = None, **config):
        """Initialize SharePoint download file tool."""
        self.name = "sharepoint_download_file"
        self.description = "Download a file from SharePoint to local storage"
        self.site_url = site_url or "https://tesco.sharepoint.com/sites/pricing"
        self.mock_mode = True
        self.logger = get_logger("tools.sharepoint.download")
        self.tracer = get_tracer("tools.sharepoint.download")
    
    def validate_parameters(self, **parameters: Any) -> bool:
        """Validate tool parameters."""
        return "file_path" in parameters
    
    async def execute(self, **parameters: Any) -> Dict[str, Any]:
        """Execute the download file operation."""
        file_path = parameters.get("file_path")
        local_path = parameters.get("local_path")
        
        with self.tracer.start_as_current_span("sharepoint_download_file_exec") as span:
            span.set_input({"file_path": file_path, "local_path": local_path})
            
            # Mock download
            if not local_path:
                temp_dir = tempfile.gettempdir()
                filename = Path(file_path).name
                local_path = str(Path(temp_dir) / filename)
            
            mock_content = f"""# Mock SharePoint Document
File: {file_path}
Downloaded from: {self.site_url}

This is mock content for local development.
"""
            
            Path(local_path).write_text(mock_content)
            
            result = {
                "success": True,
                "file_path": file_path,
                "local_path": local_path,
                "size": len(mock_content),
                "mock": True
            }
            
            span.set_output(result)
            self.logger.info(f"Downloaded {file_path} to {local_path}")
            return result


class SharePointUploadFileTool(BaseTool):
    """Tool to upload a file to SharePoint."""
    
    def __init__(self, site_url: Optional[str] = None, **config):
        """Initialize SharePoint upload file tool."""
        self.name = "sharepoint_upload_file"
        self.description = "Upload a file to SharePoint"
        self.site_url = site_url or "https://tesco.sharepoint.com/sites/pricing"
        self.mock_mode = True
        self.logger = get_logger("tools.sharepoint.upload")
        self.tracer = get_tracer("tools.sharepoint.upload")
    
    def validate_parameters(self, **parameters: Any) -> bool:
        """Validate tool parameters."""
        return "local_file_path" in parameters and "sharepoint_folder" in parameters
    
    async def execute(self, **parameters: Any) -> Dict[str, Any]:
        """Execute the upload file operation."""
        local_file_path = parameters.get("local_file_path")
        sharepoint_folder = parameters.get("sharepoint_folder")
        filename = parameters.get("filename") or Path(local_file_path).name
        overwrite = parameters.get("overwrite", True)
        
        with self.tracer.start_as_current_span("sharepoint_upload_file_exec") as span:
            span.set_input({
                "local_file_path": local_file_path,
                "sharepoint_folder": sharepoint_folder,
                "filename": filename
            })
            
            # Check if file exists
            if not Path(local_file_path).exists():
                result = {"success": False, "error": f"File not found: {local_file_path}"}
                span.set_output(result)
                return result
            
            file_size = Path(local_file_path).stat().st_size
            sharepoint_path = f"{sharepoint_folder}/{filename}"
            
            result = {
                "success": True,
                "local_file_path": local_file_path,
                "sharepoint_path": sharepoint_path,
                "size": file_size,
                "overwritten": overwrite,
                "mock": True
            }
            
            span.set_output(result)
            self.logger.info(f"Uploaded {filename} to {sharepoint_folder}")
            return result


class SharePointSearchTool(BaseTool):
    """Tool to search SharePoint documents."""
    
    def __init__(self, site_url: Optional[str] = None, **config):
        """Initialize SharePoint search tool."""
        self.name = "sharepoint_search_documents"
        self.description = "Search for documents in SharePoint by keywords"
        self.site_url = site_url or "https://tesco.sharepoint.com/sites/pricing"
        self.mock_mode = True
        self.logger = get_logger("tools.sharepoint.search")
        self.tracer = get_tracer("tools.sharepoint.search")
    
    def validate_parameters(self, **parameters: Any) -> bool:
        """Validate tool parameters."""
        return "query" in parameters
    
    async def execute(self, **parameters: Any) -> Dict[str, Any]:
        """Execute the search operation."""
        query = parameters.get("query", "")
        folder_path = parameters.get("folder_path")
        max_results = parameters.get("max_results", 10)
        
        with self.tracer.start_as_current_span("sharepoint_search_documents_exec") as span:
            span.set_input({"query": query, "folder_path": folder_path, "max_results": max_results})
            
            # Mock search results
            results = []
            
            if "basket" in query.lower():
                results.append({
                    "title": "Basket Segments Troubleshooting Guide",
                    "path": "Shared Documents/Runbooks/basket_segments_troubleshooting.pdf",
                    "summary": "Complete guide for diagnosing and resolving basket segment feed issues",
                    "modified": "2025-10-20T14:30:00Z",
                    "score": 0.89
                })
            
            if "price" in query.lower():
                results.append({
                    "title": "Price Lifecycle API Documentation",
                    "path": "Shared Documents/APIs/price_lifecycle_runbook.docx",
                    "summary": "API documentation and troubleshooting for Price Lifecycle services",
                    "modified": "2025-10-19T10:15:00Z",
                    "score": 0.76
                })
            
            results = results[:max_results]
            
            result = {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "mock": True
            }
            
            span.set_output(result)
            self.logger.info(f"SharePoint search for '{query}' found {len(results)} results")
            return result


# Factory function for tool registry
def create_sharepoint_tools(config: Optional[Dict[str, Any]] = None) -> list:
    """
    Create SharePoint tool instances for registration.
    
    Args:
        config: SharePoint configuration
    
    Returns:
        List of SharePoint tool instances
    """
    config = config or {}
    return [
        SharePointListFilesTool(**config),
        SharePointDownloadFileTool(**config),
        SharePointUploadFileTool(**config),
        SharePointSearchTool(**config)
    ]
