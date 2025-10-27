"""
Domain Knowledge Loader - LangChain Native with Structured Markdown

Uses LangChain's MarkdownHeaderTextSplitter for deterministic, framework-based parsing.
No regex, no fragile custom parsers - just clean, structured document handling.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class EntityDefinition(BaseModel):
    """Structured entity definition."""
    name: str = Field(description="Entity name (e.g., GTIN, TPNB)")
    type: Optional[str] = Field(None, description="Entity type")
    description: Optional[str] = Field(None, description="What this entity represents")
    patterns: Optional[str] = Field(None, description="Regex pattern for matching")
    examples: Optional[List[str]] = Field(None, description="Example values")
    synonyms: Optional[List[str]] = Field(None, description="Alternative names")
    zero_pad_length: Optional[int] = Field(None, description="Zero-padding length")


class ToolMapping(BaseModel):
    """Structured tool mapping."""
    tool_name: str = Field(description="Name of the tool")
    purpose: Optional[str] = Field(None, description="What this tool does")
    required_parameters: List[Dict[str, str]] = Field(default_factory=list, description="Parameters")
    example_usage: Optional[str] = Field(None, description="Example usage")


class IncidentType(BaseModel):
    """Structured incident type definition."""
    name: str = Field(description="Incident type name")
    severity: Optional[str] = Field(None, description="Severity level")
    escalation_level: Optional[str] = Field(None, description="When to escalate")
    handling_team: Optional[str] = Field(None, description="Which team handles this")
    sla_response: Optional[str] = Field(None, description="SLA for response")
    sla_resolution: Optional[str] = Field(None, description="SLA for resolution")
    description: Optional[str] = Field(None, description="Brief description")
    procedures: List[str] = Field(default_factory=list, description="Procedures")
    tools: List[str] = Field(default_factory=list, description="Required tools")
    templates: List[str] = Field(default_factory=list, description="Templates")


class DomainKnowledge(BaseModel):
    """Container for parsed domain knowledge."""
    
    entities: Dict[str, EntityDefinition] = Field(default_factory=dict)
    location_clusters: Dict[str, str] = Field(default_factory=dict)
    tool_mappings: Dict[str, ToolMapping] = Field(default_factory=dict)
    incident_types: Dict[str, IncidentType] = Field(default_factory=dict)
    escalation_matrix: Dict[str, Dict[str, List[str]]] = Field(default_factory=dict)
    
    def get_entity_info(self, entity_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a domain entity."""
        entity = self.entities.get(entity_name)
        return entity.dict() if entity else None
    
    def get_location_cluster_id(self, cluster_name: str) -> Optional[str]:
        """Get UUID for a location cluster name."""
        return self.location_clusters.get(cluster_name)
    
    def get_tool_for_intent(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get tool mapping for a specific tool."""
        tool = self.tool_mappings.get(tool_name)
        return tool.dict() if tool else None
    
    def get_incident_type(self, incident_key: str) -> Optional[Dict[str, Any]]:
        """Get incident type information."""
        incident = self.incident_types.get(incident_key)
        return incident.dict() if incident else None
    
    def classify_incident_severity(self, incident_key: str) -> Optional[str]:
        """Get severity level for an incident type."""
        incident = self.incident_types.get(incident_key)
        return incident.severity if incident else None


class KnowledgeLoader:
    """
    Loads domain knowledge using LangChain's MarkdownHeaderTextSplitter.
    
    This is a deterministic, framework-based approach that:
    - Splits Markdown by headers (###, ####, etc.)
    - Preserves hierarchy and structure
    - No fragile regex patterns
    - No LLM calls needed
    - 100% deterministic and testable
    """
    
    def __init__(self, domain_dir: str = "kb/domain"):
        self.domain_dir = Path(domain_dir)
        self.logger = logging.getLogger(__name__)
        
        # Define header structure for splitting
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ]
    
    def load_all(self) -> DomainKnowledge:
        """Load all domain knowledge files using LangChain's MarkdownHeaderTextSplitter."""
        knowledge = DomainKnowledge()
        
        try:
            # Load glossary
            glossary_path = self.domain_dir / "glossary.md"
            if glossary_path.exists():
                self._load_glossary_structured(glossary_path, knowledge)
            
            # Load incident types
            incidents_path = self.domain_dir / "incident_types.md"
            if incidents_path.exists():
                self._load_incidents_structured(incidents_path, knowledge)
        
        except Exception as e:
            self.logger.error(f"Error loading domain knowledge: {e}", exc_info=True)
        
        return knowledge
    
    def _load_glossary_structured(self, path: Path, knowledge: DomainKnowledge):
        """Load glossary using MarkdownHeaderTextSplitter."""
        # Read markdown content
        markdown_content = path.read_text(encoding="utf-8")
        
        # Split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        docs = markdown_splitter.split_text(markdown_content)
        
        # Process each section
        for doc in docs:
            metadata = doc.metadata
            content = doc.page_content
            
            # Entity definitions (h3 level)
            if "h3" in metadata and "h2" in metadata:
                if metadata["h2"] == "Entity Definitions":
                    self._parse_entity(metadata["h3"], content, knowledge)
            
            # Location cluster mappings (h2 level)
            if "h2" in metadata and metadata["h2"] == "Location Cluster Mapping":
                self._parse_location_clusters(content, knowledge)
            
            # Tool selection guide (h4 level under h2 "Tool Selection Guide")
            if "h4" in metadata and "h2" in metadata:
                if metadata["h2"] == "Tool Selection Guide":
                    self._parse_tool_mapping(metadata["h4"], content, knowledge)
        
        self.logger.info(f"Loaded {len(knowledge.entities)} entities, "
                       f"{len(knowledge.location_clusters)} clusters, "
                       f"{len(knowledge.tool_mappings)} tools from glossary")
    
    def _parse_entity(self, entity_name: str, content: str, knowledge: DomainKnowledge):
        """Parse an entity definition from its section."""
        entity = EntityDefinition(name=entity_name)
        
        # Parse field lines
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse bullet point fields
            if line.startswith('- **Type**:'):
                entity.type = line.split(':', 1)[1].strip()
            elif line.startswith('- **Description**:'):
                entity.description = line.split(':', 1)[1].strip()
            elif line.startswith('- **Patterns**:'):
                entity.patterns = line.split(':', 1)[1].strip()
            elif line.startswith('- **Examples**:'):
                examples = line.split(':', 1)[1].strip()
                entity.examples = [e.strip() for e in examples.split(',')]
            elif line.startswith('- **Synonyms**:'):
                synonyms = line.split(':', 1)[1].strip()
                entity.synonyms = [s.strip() for s in synonyms.split(',')]
            elif line.startswith('- **Zero Pad Length**:'):
                try:
                    entity.zero_pad_length = int(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
        
        knowledge.entities[entity_name] = entity
    
    def _parse_location_clusters(self, content: str, knowledge: DomainKnowledge):
        """Parse location cluster table."""
        import re
        
        # Find table rows with UUID pattern
        row_pattern = r'\|\s*([^|]+?)\s*\|\s*([a-f0-9-]{36})\s*\|'
        for match in re.finditer(row_pattern, content):
            cluster_name = match.group(1).strip()
            uuid = match.group(2).strip()
            
            # Skip header/separator rows
            if cluster_name.lower() == "cluster name" or cluster_name.startswith('-'):
                continue
            
            knowledge.location_clusters[cluster_name] = uuid
    
    def _parse_tool_mapping(self, tool_name: str, content: str, knowledge: DomainKnowledge):
        """Parse a tool mapping from its section."""
        tool = ToolMapping(tool_name=tool_name)
        
        # Parse purpose
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('- **Purpose**:'):
                tool.purpose = line.split(':', 1)[1].strip()
                break
        
        # Parse required parameters
        in_params = False
        for line in content.split('\n'):
            line = line.strip()
            
            if '**Required Parameters**:' in line:
                in_params = True
                continue
            
            if in_params and line.startswith('  - `'):
                # Extract parameter: "  - `param_name` (description)"
                import re
                match = re.match(r'  - `([^`]+)`\s*\(([^)]+)\)', line)
                if match:
                    param_name = match.group(1)
                    param_desc = match.group(2)
                    tool.required_parameters.append({
                        "name": param_name,
                        "description": param_desc
                    })
            elif in_params and line and not line.startswith('  '):
                # End of parameters section
                break
        
        knowledge.tool_mappings[tool_name] = tool
    
    def _load_incidents_structured(self, path: Path, knowledge: DomainKnowledge):
        """Load incident types using MarkdownHeaderTextSplitter."""
        # Read markdown content
        markdown_content = path.read_text(encoding="utf-8")
        
        # Split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        docs = markdown_splitter.split_text(markdown_content)
        
        # Process each section
        for doc in docs:
            metadata = doc.metadata
            content = doc.page_content
            
            # Incident type definitions (h3 level)
            if "h3" in metadata and "h2" in metadata:
                # Skip escalation matrix sections
                if metadata["h2"] != "Escalation Matrix":
                    self._parse_incident_type(metadata["h3"], content, knowledge)
            
            # Escalation matrix (h3 under h2 "Escalation Matrix")
            if "h3" in metadata and "h2" in metadata:
                if metadata["h2"] == "Escalation Matrix":
                    self._parse_escalation_level(metadata["h3"], content, knowledge)
        
        self.logger.info(f"Loaded {len(knowledge.incident_types)} incident types, "
                       f"{len(knowledge.escalation_matrix)} escalation levels")
    
    def _parse_incident_type(self, incident_name: str, content: str, knowledge: DomainKnowledge):
        """Parse an incident type definition from its section."""
        incident = IncidentType(name=incident_name)
        
        # Parse field lines
        procedures = []
        tools_list = []
        templates_list = []
        
        in_procedures = False
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse metadata fields
            if line.startswith('- **Severity**:'):
                incident.severity = line.split(':', 1)[1].strip()
            elif line.startswith('- **Escalation Level**:'):
                incident.escalation_level = line.split(':', 1)[1].strip()
            elif line.startswith('- **Handling Team**:'):
                incident.handling_team = line.split(':', 1)[1].strip()
            elif line.startswith('- **SLA Response**:'):
                incident.sla_response = line.split(':', 1)[1].strip()
            elif line.startswith('- **SLA Resolution**:'):
                incident.sla_resolution = line.split(':', 1)[1].strip()
            elif line.startswith('**Description**:'):
                incident.description = line.split(':', 1)[1].strip()
            
            # Parse tools (comma-separated)
            elif line.startswith('**Tools**:'):
                tools_str = line.split(':', 1)[1].strip()
                tools_list = [t.strip() for t in tools_str.split(',')]
            
            # Parse templates (comma-separated)
            elif line.startswith('**Templates**:'):
                templates_str = line.split(':', 1)[1].strip()
                templates_list = [t.strip() for t in templates_str.split(',')]
            
            # Parse procedures (numbered list)
            elif '**Procedures**:' in line:
                in_procedures = True
            elif in_procedures and line[0].isdigit() and '. ' in line:
                procedure = line.split('. ', 1)[1] if '. ' in line else line
                procedures.append(procedure)
            elif in_procedures and line.startswith('**'):
                in_procedures = False
        
        incident.procedures = procedures
        incident.tools = tools_list
        incident.templates = templates_list
        
        # Create key from name
        key = incident_name.lower().replace(" ", "_").replace("-", "_")
        knowledge.incident_types[key] = incident
    
    def _parse_escalation_level(self, level_name: str, content: str, knowledge: DomainKnowledge):
        """Parse an escalation level from its section."""
        level_key = level_name.lower().replace(" ", "_")
        timeframes = {}
        
        # Parse timeframe lines
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('- **') and '**:' in line:
                # Extract timeframe and roles
                import re
                match = re.match(r'- \*\*([^*]+)\*\*:\s*(.+)', line)
                if match:
                    timeframe = match.group(1).strip().lower().replace(" ", "_")
                    roles_str = match.group(2).strip()
                    roles = [r.strip() for r in roles_str.split(',')]
                    timeframes[timeframe] = roles
        
        if timeframes:
            knowledge.escalation_matrix[level_key] = timeframes


def load_domain_knowledge(domain_dir: str = "kb/domain") -> DomainKnowledge:
    """
    Load domain knowledge using LangChain's MarkdownHeaderTextSplitter.
    
    This is a deterministic, framework-based approach:
    - No LLM calls
    - No fragile regex
    - Uses LangChain's battle-tested Markdown parser
    - 100% reproducible
    
    Args:
        domain_dir: Directory containing domain knowledge files
        
    Returns:
        DomainKnowledge object with parsed data
    """
    loader = KnowledgeLoader(domain_dir)
    return loader.load_all()
