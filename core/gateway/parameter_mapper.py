"""
Automatic parameter mapping for tool execution.

Maps extracted entities to tool parameters using JSON Schema validation.
Industry standard approach - deterministic, no LLM hallucination risk.
"""

import json
import re
from typing import Any, Dict, List, Optional


class ParameterMapper:
    """
    Maps entities to tool parameters based on JSON Schema.
    
    Industry best practice: Use schema-driven mapping instead of LLM for
    parameter formation to avoid hallucinations and ensure correctness.
    """

    def __init__(self):
        """Initialize parameter mapper."""
        self.entity_to_param_map = {
            "gtin": ["gtin", "gTin", "productId"],
            "tpnb": ["tpnb", "tpnb", "TPNB", "productCode"],
            "locations": ["location", "locationClusterId", "locations", "store"],
            "key_terms": ["query", "search", "keywords"],
        }
    
    def map_parameters(
        self, 
        tool_name: str, 
        tool_schema: Dict[str, Any], 
        entities: Dict[str, Any],
        ticket_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Automatically map entities to tool parameters using schema.
        
        Args:
            tool_name: Name of the tool
            tool_schema: JSON Schema for tool input parameters
            entities: Extracted entities (GTIN, TPNB, etc.)
            ticket_context: Optional ticket context (title, description, etc.)
        
        Returns:
            Formed parameters dict ready for tool execution
        """
        ticket_context = ticket_context or {}
        parameters = {}
        
        # Get schema properties
        properties = tool_schema.get("properties", {}) if isinstance(tool_schema, dict) else {}
        required = tool_schema.get("required", []) if isinstance(tool_schema, dict) else []
        
        # Map entities to parameters using schema
        for param_name, param_schema in properties.items():
            param_type = param_schema.get("type", "string")
            param_default = param_schema.get("default")
            
            # Try to find entity value for this parameter
            value = self._find_entity_value(param_name, entities, ticket_context)
            
            # If not found but has default, use default
            if value is None and param_default is not None:
                value = param_default
            
            # Apply type conversion if needed
            if value is not None:
                value = self._convert_type(value, param_type)
                parameters[param_name] = value
        
        # Special handling for common tools (can be extended)
        parameters = self._apply_tool_specific_logic(tool_name, parameters, entities, ticket_context)
        
        # Validate required parameters
        missing_required = [p for p in required if p not in parameters]
        if missing_required:
            # Try to fill with sensible defaults
            for param in missing_required:
                parameters[param] = self._get_default_value(param, tool_name, entities)
        
        return parameters
    
    def _find_entity_value(
        self, 
        param_name: str, 
        entities: Dict[str, Any],
        ticket_context: Dict[str, Any]
    ) -> Any:
        """Find entity value that matches this parameter name."""
        param_lower = param_name.lower()
        
        # Direct match
        if param_name in entities:
            return entities[param_name]
        
        # Check entity_to_param_map
        for entity_key, param_aliases in self.entity_to_param_map.items():
            if param_lower in [p.lower() for p in param_aliases]:
                entity_value = entities.get(entity_key)
                if entity_value:
                    # Handle arrays (e.g., locations)
                    if isinstance(entity_value, list) and len(entity_value) > 0:
                        return entity_value[0] if param_type != "array" else entity_value
                    return entity_value
        
        # Check if param name matches entity keys
        for entity_key in entities.keys():
            if param_lower == entity_key.lower() or param_lower in entity_key.lower():
                value = entities[entity_key]
                if isinstance(value, list) and len(value) > 0:
                    return value[0]
                return value
        
        # Special handling for query/search parameters
        if param_name.lower() in ["query", "search", "nrql"]:
            key_terms = entities.get("key_terms", [])
            if key_terms:
                return " OR ".join(key_terms[:5])  # Limit to 5 terms
            # Fallback to ticket title/description
            return ticket_context.get("title", "")[:100]
        
        return None
    
    def _convert_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type."""
        if target_type == "string":
            return str(value)
        elif target_type == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return 0
        elif target_type == "number":
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        elif target_type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes")
        elif target_type == "array":
            if isinstance(value, list):
                return value
            return [value] if value else []
        return value
    
    def _apply_tool_specific_logic(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        entities: Dict[str, Any],
        ticket_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply tool-specific parameter logic."""
        # Splunk search: build query from key_terms
        if tool_name == "splunk_search" and "query" not in parameters:
            key_terms = entities.get("key_terms", [])
            incident_type = entities.get("incident_type", "")
            
            if key_terms:
                # Build OR query from key terms
                query_parts = [f'"{term}"' for term in key_terms[:3]]
                parameters["query"] = " OR ".join(query_parts)
            elif incident_type:
                parameters["query"] = incident_type.replace("_", " ")
            else:
                title = ticket_context.get("title", "")
                parameters["query"] = title[:50] if title else "*"
            
            # Set default time_range if not provided
            if "time_range" not in parameters:
                parameters["time_range"] = "-2h"
            
            # Set default index if not provided
            if "index" not in parameters:
                parameters["index"] = "price-advisory-*"
        
        # New Relic: build NRQL query
        elif tool_name == "newrelic_metrics" or tool_name == "newrelic_query":
            if "nrql" not in parameters:
                key_terms = entities.get("key_terms", [])
                query = "SELECT average(duration), count(*) FROM Transaction"
                if key_terms:
                    query += f" WHERE name LIKE '%{key_terms[0]}%'"
                query += " SINCE 1 hour ago"
                parameters["nrql"] = query
        
        # SharePoint: use archive folder by default
        elif tool_name.startswith("sharepoint_"):
            if "folder_path" not in parameters and "file_path" not in parameters:
                incident_type = entities.get("incident_type", "")
                if "competitor" in incident_type.lower():
                    parameters["folder_path"] = "Shared Documents/Archive/Competitor"
                elif "basket" in incident_type.lower():
                    parameters["folder_path"] = "Shared Documents/Archive/Basket"
                else:
                    parameters["folder_path"] = "Shared Documents/Archive"
        
        return parameters
    
    def _get_default_value(
        self, 
        param: str, 
        tool_name: str, 
        entities: Dict[str, Any]
    ) -> Any:
        """Get sensible default value for missing required parameter."""
        defaults = {
            "time_range": "-1h",
            "max_results": 100,
            "locationClusterId": "default",
            "account_id": "default",
            "index": "price-advisory-*",
        }
        
        return defaults.get(param, "")

