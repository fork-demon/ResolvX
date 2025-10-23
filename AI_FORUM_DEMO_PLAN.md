# 🎯 AI Forum Demo Plan: Golden Agent Framework

## 📋 **Presentation Overview**

**Duration**: 45-60 minutes  
**Audience**: AI Forum (Technical)  
**Focus**: Agentic Design Patterns, Architecture, and Live Demo  

---

## 🎬 **Part 1: Introduction & Architecture (15 minutes)**

### **1.1 Framework Overview (5 min)**
```
┌─────────────────────────────────────────────────────────────────┐
│                    GOLDEN AGENT FRAMEWORK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🎯 MISSION: Production-ready multi-agent system for           │
│      enterprise incident response and automation               │
│                                                                 │
│  🏗️ ARCHITECTURE: LangGraph + MCP + RAG + Evaluation          │
│                                                                 │
│  🔧 COMPONENTS:                                                │
│     • 4 Core Agents (Poller, Triage, Memory, Supervisor)       │
│     • 12+ MCP Tools (Splunk, NewRelic, Price APIs)            │
│     • RAG System (Local KB + FAISS Memory)                     │
│     • Evaluation System (Guardrails, Hallucination, Quality)   │
│     • Observability (LangFuse tracing)                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### **1.2 Design Patterns Overview (10 min)**

#### **🔧 Core Design Patterns**

**1. Agent Pattern (BaseAgent)**
```python
class BaseAgent(ABC):
    def __init__(self, name: str, config: AgentConfig):
        self.name = name
        self.config = config
        self.logger = get_logger(f"agent.{name}")
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        pass
```

**2. ToolProxy Pattern (MCP Gateway)**
```python
class ToolRegistry:
    def __init__(self, mcp_client: MCPClient):
        self._mcp_proxy = MCPToolProxy(mcp_client)
    
    async def call_tool(self, tool_name: str, params: Dict):
        return await self._mcp_proxy.execute(tool_name, params)
```

**3. Strategy Pattern (Agent Behavior)**
```python
# Behavior driven by prompts, not hard-coded logic
prompts:
  system_prompt:
    type: "file"
    path: "prompts/triage/system_prompt.md"
```

**4. Observer Pattern (Evaluation)**
```python
# Evaluation components observe agent outputs
self.guardrails = Guardrails(config.guardrails)
self.hallucination_checker = HallucinationChecker(config.hallucination)
self.quality_assessor = QualityAssessor(config.quality)
```

---

## 🧠 **Part 2: Agentic Design Patterns (20 minutes)**

### **2.1 ReAct Pattern Implementation (8 min)**

#### **Triage Agent ReAct Flow**
```
┌─────────────────────────────────────────────────────────────────┐
│                        REACT PATTERN                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🤔 REASONING:                                                  │
│     • RAG Search → Find similar incidents                      │
│     • LLM Analysis → Extract entities, determine severity      │
│     • Tool Selection → Choose appropriate tools               │
│                                                                 │
│  🛠️ ACTION:                                                    │
│     • Execute Splunk queries                                  │
│     • Query New Relic metrics                                 │
│     • Call Price APIs                                         │
│                                                                 │
│  📊 OBSERVATION:                                               │
│     • Analyze tool results                                    │
│     • Update context                                          │
│     • Make routing decisions                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### **Code Implementation**
```python
async def _analyze_incident(self, state: AgentState) -> AgentState:
    # REASONING: RAG + LLM Analysis
    similar_incidents = self.rag.search(combined_text, k=3)
    llm_analysis = await self._get_llm_analysis(ticket, runbook_guidance)
    
    # ACTION: Tool Selection & Execution
    tools_to_call = self._extract_tools_from_llm(llm_analysis)
    for tool_name in tools_to_call:
        result = await self.tool_registry.call_tool(tool_name, params)
    
    # OBSERVATION: Result Analysis
    return self._synthesize_results(tool_results, llm_analysis)
```

### **2.2 Multi-Agent Orchestration (6 min)**

#### **LangGraph State Machine**
```python
def build_graph(self) -> StateGraph:
    graph = StateGraph(AgentState)
    
    # Add nodes (agents)
    graph.add_node("analyze_incident", self._analyze_incident)
    graph.add_node("determine_severity", self._determine_severity)
    graph.add_node("route_incident", self._route_incident)
    
    # Add edges (flow)
    graph.add_edge("analyze_incident", "determine_severity")
    graph.add_edge("determine_severity", "route_incident")
    
    return graph.compile()
```

#### **Agent Communication Flow**
```
Poller → Triage → Memory → Supervisor
  ↓        ↓        ↓         ↓
Tickets  Analysis  Duplicates  Decision
```

### **2.3 Tool Integration Patterns (6 min)**

#### **MCP Tool Pattern**
```python
# Tools are discovered dynamically
tools:
  splunk_search:
    type: "mcp"
    server: "splunk-mcp-server"
    description: "Search Splunk logs and metrics"
  
  newrelic_metrics:
    type: "mcp"
    server: "newrelic-mcp-server"
    description: "Query New Relic metrics"
```

#### **Tool Execution with Circuit Breaker**
```python
async def execute_tool(self, tool_name: str, params: Dict):
    with self.tracer.start_as_current_span(f"tool_{tool_name}"):
        result = await with_circuit_breaker(
            name="mcp_gateway",
            func=self._execute_tool_request,
            tool_name=tool_name,
            parameters=params,
            fallback=self._execute_local_fallback
        )
        return result
```

---

## 🎮 **Part 3: Live Demo (20 minutes)**

### **3.1 Setup & Architecture Demo (5 min)**

#### **Start Services**
```bash
# Start Ollama LLM
ollama serve

# Start MCP Mock Gateway
docker-compose up -d

# Start LangFuse
docker-compose up -d langfuse
```

#### **Show Architecture**
- **4 Agents**: Poller, Triage, Memory, Supervisor
- **12 Tools**: Splunk, NewRelic, Price APIs, etc.
- **RAG System**: Local KB + FAISS Memory
- **Evaluation**: Guardrails, Hallucination, Quality

### **3.2 End-to-End Workflow Demo (10 min)**

#### **Step 1: Poller Agent**
```bash
# Show ticket polling
python scripts/test_full_workflow.py
```

**What to Show:**
- Poller discovers 5 tickets from MCP gateway
- Tickets forwarded to Triage agent
- LangFuse trace: `poller_process` span

#### **Step 2: Triage Agent (ReAct Pattern)**
**REASONING:**
- RAG search finds similar incidents
- LLM analyzes ticket content
- Tool selection based on LLM recommendations

**ACTION:**
- Execute Splunk queries
- Query New Relic metrics
- Call Price APIs

**OBSERVATION:**
- Synthesize results
- Make routing decisions

#### **Step 3: Memory Agent**
- Check for duplicate tickets
- FAISS vector search (90% similarity)
- Store new tickets or forward duplicates

#### **Step 4: Supervisor Agent**
- Receive enriched data
- Make final decisions
- Route to appropriate teams

### **3.3 LangFuse Tracing Demo (5 min)**

#### **Show Trace Structure**
```
triage_process
├── rag_knowledge_search
│   ├── input: {"query": "basket segments failure"}
│   └── output: {"results": [...]}
├── llm_chat_completion
│   ├── input: {"messages": [...]}
│   └── output: {"content": "{\"tools_to_use\": [...]}"}
├── guardrails_evaluation
│   ├── input: {"content": "LLM response"}
│   └── output: {"passed": true, "violations": []}
├── hallucination_evaluation
│   ├── input: {"response": "LLM response"}
│   └── output: {"has_hallucination": false}
├── quality_evaluation
│   ├── input: {"response": "LLM response"}
│   └── output: {"overall_score": 0.85}
└── mcp_tool_splunk_search
    ├── input: {"query": "CreateBasketSegmentsProcessor"}
    └── output: {"results": [...]}
```

#### **Show Evaluation Metrics**
- Guardrail violations: 0
- Hallucination detections: 1
- Quality scores: 0.85 average
- Tool execution times: <2s each

---

## 🎯 **Part 4: Advanced Features (10 minutes)**

### **4.1 RAG System Deep Dive (5 min)**

#### **Local KB RAG**
```python
# Knowledge base ingestion
kb = LocalKB(knowledge_dir="kb", model_name="all-MiniLM-L6-v2")
kb.load()  # Auto-indexes all .md files

# Semantic search
results = kb.search("basket segments failure", k=3)
# Returns: [{"path": "basket_segments_runbook.md", "score": 0.92, "text": "..."}]
```

#### **FAISS Memory RAG**
```python
# Vector similarity search
memory = FAISSMemory(config={
    "index_type": "IndexFlatIP",
    "metric": "IP",
    "dimension": 384
})

# Duplicate detection
similar_tickets = memory.search(ticket_content, threshold=0.9)
```

### **4.2 Evaluation System (5 min)**

#### **Guardrails**
```python
guardrail_result = await self.guardrails.check_content(
    content=llm_analysis,
    content_type="json",
    context={"ticket_id": ticket_id}
)
# Checks: PII detection, safety checks, policy enforcement
```

#### **Hallucination Detection**
```python
hallucination_result = await self.hallucination_checker.check_response(
    response=llm_analysis,
    context={"ticket": ticket, "runbook_guidance": runbook_guidance},
    sources=similar_incidents
)
# Detects: factual errors, fabricated citations, speculative claims
```

#### **Quality Assessment**
```python
quality_result = await self.quality_assessor.assess_response(
    response=llm_analysis,
    context={"ticket": ticket, "tools_recommended": tools_to_call},
    expected_format="json"
)
# Evaluates: accuracy, completeness, relevance, coherence
```

---

## 🚀 **Part 5: Q&A & Discussion (10 minutes)**

### **5.1 Key Questions to Address**

#### **Architecture Questions**
- **Q**: "How does this scale to production?"
- **A**: MCP gateway, circuit breakers, observability, evaluation

- **Q**: "What about security and compliance?"
- **A**: Guardrails, PII detection, audit trails, role-based access

#### **Technical Questions**
- **Q**: "How do you handle tool failures?"
- **A**: Circuit breakers, fallback mechanisms, graceful degradation

- **Q**: "What about LLM hallucinations?"
- **A**: Multi-layer evaluation, fact-checking, source verification

#### **Business Questions**
- **Q**: "How does this reduce incident response time?"
- **A**: Automated triage, intelligent routing, historical context

- **Q**: "What's the ROI?"
- **A**: Faster resolution, reduced MTTR, improved quality

### **5.2 Demo Scenarios**

#### **Scenario 1: Critical Incident**
- High severity ticket
- Multiple tool calls
- Escalation to supervisor
- Human-in-the-loop

#### **Scenario 2: Duplicate Detection**
- Similar ticket exists
- Memory agent finds duplicate
- Returns resolution
- Prevents duplicate work

#### **Scenario 3: Tool Failure**
- Splunk API down
- Circuit breaker activates
- Fallback to local data
- Graceful degradation

---

## 📊 **Demo Preparation Checklist**

### **Pre-Demo Setup**
- [ ] Ollama running with `llama3.2`
- [ ] Docker Compose services up
- [ ] LangFuse accessible at `localhost:3000`
- [ ] Mock tickets in `data/mock_tickets/`
- [ ] Knowledge base files in `kb/`

### **Demo Scripts**
- [ ] `scripts/test_full_workflow.py` - End-to-end demo
- [ ] `scripts/test_memory_duplicate.py` - Memory agent demo
- [ ] `scripts/test_evaluation_integration.py` - Evaluation demo

### **Backup Plans**
- [ ] Screenshots of LangFuse traces
- [ ] Pre-recorded demo video
- [ ] Fallback to architecture diagrams

---

## 🎯 **Key Takeaways**

### **1. Design Patterns**
- **Agent Pattern**: Encapsulated behavior with state
- **ToolProxy Pattern**: Unified tool access via MCP
- **Strategy Pattern**: Prompt-driven behavior
- **Observer Pattern**: Evaluation and monitoring

### **2. Agentic Patterns**
- **ReAct**: Reasoning → Action → Observation
- **Multi-Agent**: Coordinated workflows
- **Tool Integration**: Dynamic tool discovery
- **Evaluation**: Continuous quality assessment

### **3. Production Readiness**
- **Observability**: Full tracing with LangFuse
- **Reliability**: Circuit breakers, fallbacks
- **Security**: Guardrails, PII detection
- **Scalability**: MCP gateway, async processing

### **4. Business Value**
- **Automation**: Reduces manual triage
- **Intelligence**: Context-aware decisions
- **Quality**: Evaluation prevents errors
- **Speed**: Faster incident response

---

## 📝 **Presentation Notes**

### **Opening Hook**
"Imagine a system that can automatically triage incidents, call the right tools, and make intelligent routing decisions - all while learning from historical patterns and preventing hallucinations. That's what we've built."

### **Key Messages**
1. **Not just another chatbot** - This is a production-ready multi-agent system
2. **Real agentic patterns** - ReAct, tool integration, evaluation
3. **Enterprise-grade** - Observability, security, reliability
4. **Live demo** - See it work with real tickets and tools

### **Closing**
"This framework demonstrates how to build production-ready agentic systems that can handle real-world complexity while maintaining quality and reliability. The patterns we've shown can be applied to any domain requiring intelligent automation."

---

**🎯 Ready to demo! The framework is production-ready with comprehensive tracing, evaluation, and real-world tool integration.**
