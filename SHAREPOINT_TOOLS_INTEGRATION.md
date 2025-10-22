# SharePoint Tools Integration ✅

## Overview

SharePoint tools have been successfully integrated as **local tools** (non-MCP) that are automatically discovered by all agents.

## Available SharePoint Tools

### 1. `sharepoint_list_files`
**Description**: List files and folders in a SharePoint directory

**Parameters**:
- `folder_path` (required): SharePoint folder path (e.g., "Shared Documents/Runbooks")
- `recursive` (optional): List files recursively in subfolders
- `file_extension` (optional): Filter by file extension (e.g., "pdf", "docx", "md")

**Example**:
```python
result = await tool_registry.call_tool("sharepoint_list_files", {
    "folder_path": "Shared Documents/Runbooks",
    "file_extension": "pdf"
})
```

**Returns**:
```json
{
  "success": true,
  "folder_path": "Shared Documents/Runbooks",
  "files": [
    {
      "name": "basket_segments_troubleshooting.pdf",
      "path": "Shared Documents/Runbooks/basket_segments_troubleshooting.pdf",
      "size": 245678,
      "modified": "2025-10-20T14:30:00Z",
      "type": "file",
      "extension": "pdf"
    }
  ],
  "count": 1,
  "mock": true
}
```

---

### 2. `sharepoint_download_file`
**Description**: Download a file from SharePoint to local storage

**Parameters**:
- `file_path` (required): SharePoint file path to download
- `local_path` (optional): Local path to save the file (uses temp dir if not provided)

**Example**:
```python
result = await tool_registry.call_tool("sharepoint_download_file", {
    "file_path": "Shared Documents/Runbooks/basket_segments.pdf",
    "local_path": "/tmp/runbook.pdf"
})
```

**Returns**:
```json
{
  "success": true,
  "file_path": "Shared Documents/Runbooks/basket_segments.pdf",
  "local_path": "/tmp/runbook.pdf",
  "size": 12345,
  "mock": true
}
```

---

### 3. `sharepoint_upload_file`
**Description**: Upload a file to SharePoint

**Parameters**:
- `local_file_path` (required): Local file path to upload
- `sharepoint_folder` (required): Target SharePoint folder path
- `filename` (optional): Target filename in SharePoint
- `overwrite` (optional): Overwrite if file exists (default: true)

**Example**:
```python
result = await tool_registry.call_tool("sharepoint_upload_file", {
    "local_file_path": "/tmp/analysis_report.pdf",
    "sharepoint_folder": "Shared Documents/Reports",
    "filename": "ticket_ALERT-001_analysis.pdf"
})
```

**Returns**:
```json
{
  "success": true,
  "local_file_path": "/tmp/analysis_report.pdf",
  "sharepoint_path": "Shared Documents/Reports/ticket_ALERT-001_analysis.pdf",
  "size": 54321,
  "mock": true
}
```

---

### 4. `sharepoint_search_documents`
**Description**: Search for documents in SharePoint by keywords

**Parameters**:
- `query` (required): Search query keywords
- `folder_path` (optional): Limit search to specific folder
- `max_results` (optional): Maximum number of results (default: 10)

**Example**:
```python
result = await tool_registry.call_tool("sharepoint_search_documents", {
    "query": "basket segments troubleshooting",
    "max_results": 5
})
```

**Returns**:
```json
{
  "success": true,
  "query": "basket segments troubleshooting",
  "results": [
    {
      "title": "Basket Segments Troubleshooting Guide",
      "path": "Shared Documents/Runbooks/basket_segments_troubleshooting.pdf",
      "summary": "Complete guide for diagnosing and resolving basket segment feed issues",
      "modified": "2025-10-20T14:30:00Z",
      "score": 0.89
    }
  ],
  "count": 1,
  "mock": true
}
```

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tool Registry (Unified Interface)                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Local Tools (Direct Execution):                            │
│  ├─ SharePointListFilesTool                                 │
│  ├─ SharePointDownloadFileTool                              │
│  ├─ SharePointUploadFileTool                                │
│  └─ SharePointSearchTool                                    │
│                                                              │
│  MCP Tools (via Gateway):                                   │
│  ├─ splunk_search                                           │
│  ├─ newrelic_metrics                                        │
│  ├─ base_prices_get                                         │
│  └─ ... (9 more tools)                                      │
│                                                              │
│  All tools accessible via:                                  │
│  await tool_registry.call_tool(tool_name, params)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Agent Discovery

1. **Automatic Registration**:
   - SharePoint tools are automatically loaded when ToolRegistry initializes
   - No configuration needed - works out of the box

2. **Agent Access**:
   ```python
   # Agents can list all available tools
   all_tools = tool_registry.list_tools()
   # Returns: ['sharepoint_list_files', 'splunk_search', 'newrelic_metrics', ...]
   ```

3. **LLM-Driven Selection**:
   - The LLM receives the complete list of available tools (including SharePoint)
   - LLM can recommend SharePoint tools based on ticket analysis
   - Example: "Download related runbooks from SharePoint for reference"

### Workflow Integration

```
Ticket arrives → Triage Agent
  │
  ├─ RAG searches local KB
  │
  ├─ LLM analyzes ticket + runbook
  │  └─ LLM sees: splunk_search, newrelic_metrics, sharepoint_*, etc.
  │  └─ LLM recommends: ["splunk_search", "sharepoint_search_documents"]
  │
  ├─ Execute splunk_search (via MCP)
  ├─ Execute sharepoint_search_documents (local)
  │
  └─ Forward enriched data to Supervisor
```

## Mock vs Production Mode

### Current: Mock Mode (Local Development)
- **Enabled by default** - no credentials needed
- Returns simulated SharePoint data
- Perfect for testing and development

### Production Mode
To enable real SharePoint integration, configure credentials in `.env`:

```bash
SHAREPOINT_SITE_URL=https://tesco.sharepoint.com/sites/pricing
SHAREPOINT_CLIENT_ID=your-azure-ad-app-id
SHAREPOINT_CLIENT_SECRET=your-azure-ad-secret
SHAREPOINT_TENANT_ID=your-tenant-id
```

## Tracing

All SharePoint tool executions are fully traced in Langfuse with:
- ✅ Input parameters (folder_path, query, etc.)
- ✅ Output results (files found, download status, etc.)
- ✅ Timing and performance metrics
- ✅ Parent-child relationships with Triage agent

**Span names**:
- `sharepoint_list_files_exec`
- `sharepoint_download_file_exec`
- `sharepoint_upload_file_exec`
- `sharepoint_search_documents_exec`

## Use Cases

### 1. Download Additional Runbooks
If the local KB doesn't have enough information, the LLM can recommend downloading related runbooks from SharePoint.

### 2. Upload Analysis Reports
After completing analysis, the agent can upload a summary report to SharePoint for team review.

### 3. Search for Related Documentation
Find related incident reports, API documentation, or troubleshooting guides stored in SharePoint.

### 4. List Available Resources
Check what runbooks and documentation are available for a specific service or component.

## Testing

Run the full workflow test to see SharePoint tools in action:

```bash
python scripts/test_full_workflow.py
```

View traces in Langfuse: http://localhost:3000

## Implementation Details

**File**: `core/tools/sharepoint_tool.py`  
**Classes**:
- `SharePointListFilesTool(BaseTool)`
- `SharePointDownloadFileTool(BaseTool)`
- `SharePointUploadFileTool(BaseTool)`
- `SharePointSearchTool(BaseTool)`

**Registration**: Automatic via `ToolRegistry._load_local_tools()`

**Execution**: Same interface as MCP tools - `await tool_registry.call_tool(name, params)`

---

✅ **SharePoint tools are production-ready and fully integrated!**

