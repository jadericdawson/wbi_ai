# Multi-Scratchpad Collaborative Agent System

## Overview
Agents collaborate on **7 specialized scratchpads**, using **line-level editing** (like Claude Code) to efficiently build outputs without rewriting entire documents.

## Scratchpad Types

| Pad Name | Purpose | Typical Users | Operations |
|----------|---------|---------------|------------|
| **output** | Final answer assembly | Writer, Engineer, Validator | Line-level edits, incremental building |
| **research** | Raw findings, search results | Tool Agent, Query Refiner | Append findings as they're discovered |
| **tables** | Formatted markdown tables | Tool Agent, Writer | Create/store tables separately |
| **plots** | Plot data and specifications | Tool Agent | Store plot configs, data arrays |
| **outline** | Initial plan and structure | Writer, Orchestrator | Create roadmap, track progress |
| **data** | Structured JSON/lists | Tool Agent | Raw data before formatting |
| **log** | Agent actions and workflow | All agents | Track decisions and delegation |

## Key Operations

### Line-Level Editing (Efficient - No Full Rewrites)
```python
# Insert at specific line (1-indexed)
scratchpad_insert_lines("output", "introduction", 5, "New content here")

# Replace line range
scratchpad_replace_lines("output", "intro", 3, 5, "Replacement content")

# Delete lines
scratchpad_delete_lines("output", "intro", 10, 12)

# View with line numbers
scratchpad_get_lines("output", "intro", 1, 20)
```

### Section-Level Operations
```python
# Write/append/prepend entire sections
scratchpad_write("research", "findings", content, mode="append")

# Read sections
scratchpad_read("tables", "metrics_table")

# Find/replace text
scratchpad_edit("output", "intro", "old text", "new text")

# List all sections in a pad
scratchpad_list("research")

# Get overview of all pads
scratchpad_summary()
```

## Example Workflow

### Phase 1: Planning (Loop 1)
**Orchestrator** delegates to **Writer**:
```
Task: "Create an outline in the OUTLINE pad"
```

**Writer** executes:
```python
scratchpad_write("outline", "plan", """
1. Introduction
2. Data Analysis
3. Results Table
4. Conclusion
""")
```

### Phase 2: Data Gathering (Loops 2-5, Parallel)
**Orchestrator** delegates to **Tool Agent**:
```
Task: "Search for performance metrics and save to RESEARCH pad"
```

**Tool Agent** executes:
```python
# Search
results = search_knowledge_base(["performance", "metrics"], "...", 10)

# Save findings
scratchpad_write("research", "perf_data", 
                 f"Found {len(results)} documents:\n{results}", 
                 mode="append")

# Create table
table = to_markdown_table(parsed_metrics)
scratchpad_write("tables", "perf_metrics", table)
```

### Phase 3: Building Output (Loops 6-8)
**Writer** reads and assembles:
```python
# Read research findings
research = scratchpad_read("research", "perf_data")

# Start building output
scratchpad_write("output", "introduction", """
# Performance Analysis Report

This report analyzes system performance metrics.
""")

# Add key findings incrementally
scratchpad_insert_lines("output", "introduction", -1, 
                        "Key Finding: 42% improvement observed")

# Pull in table from TABLES pad
table = scratchpad_read("tables", "perf_metrics")
scratchpad_write("output", "results", f"""
## Results

{table}
""")
```

### Phase 4: Technical Details (Loop 9)
**Engineer** adds technical analysis:
```python
# Check what Writer has so far
current = scratchpad_get_lines("output", "results")

# Add technical note at specific location
scratchpad_insert_lines("output", "results", 5, 
                        "Technical Note: Latency reduced from 200ms to 116ms")
```

### Phase 5: Final Review and Edit (Loop 10)
**Writer** reviews and refines:
```python
# View current output
scratchpad_get_lines("output", "introduction")

# Make precise edits
scratchpad_replace_lines("output", "introduction", 4, 4, 
                         "Key Finding: 42.3% improvement observed in production")

# Add conclusion
scratchpad_write("output", "conclusion", """
## Conclusion

The analysis demonstrates significant performance gains...
""")
```

### Phase 6: Validation (Loop 11)
**Validator** checks:
```python
# Read entire output pad
final_output = scratchpad_read("output")

# Verify against user's original question in LOG pad
user_goal = scratchpad_read("log", "user_goal")

# Approve or request changes
return {"response": "VALIDATION_PASSED - Output complete"}
```

### Phase 7: Delivery
**System** returns OUTPUT pad content to user as final answer.

## Benefits

### 1. Efficiency
- No need to rewrite entire documents
- Line-level operations like Claude Code
- Only transmit changed lines, not full content

### 2. Parallel Collaboration
- Tool Agent works on RESEARCH while Writer plans OUTLINE
- Multiple agents edit different sections simultaneously
- Tables/plots built separately, then integrated

### 3. Incremental Building
- Start with outline, fill in sections progressively
- Add details as research completes
- Review and refine specific lines without rebuilding

### 4. Clear Separation of Concerns
- Research findings don't clutter final output
- Tables stored separately, reusable
- Workflow tracked independently in LOG pad

### 5. Async Multi-Agent Coordination
```
Loop 3: Tool Agent → writes to RESEARCH pad
Loop 3: Writer → reads OUTLINE, starts OUTPUT intro
Loop 4: Tool Agent → creates table in TABLES pad  
Loop 5: Writer → reads RESEARCH, adds to OUTPUT
Loop 6: Engineer → inserts technical detail in OUTPUT line 15
Loop 7: Writer → edits OUTPUT line 15, adds conclusion
```

## Agent Responsibilities

| Agent | Primary Pads | Operations |
|-------|-------------|------------|
| **Orchestrator** | log | Track workflow, delegate tasks |
| **Tool Agent** | research, tables, plots, data | Search, format, store data |
| **Writer** | outline, output | Plan structure, assemble final answer |
| **Engineer** | output, data | Add technical analysis |
| **Query Refiner** | log, research | Analyze search quality |
| **Validator** | output | Final quality check |
| **Supervisor** | all (read-only) | Evaluate completion readiness |

## Tool Signatures

```python
# Reading
scratchpad_summary() -> str
scratchpad_list(pad_name: str) -> str
scratchpad_read(pad_name: str, section_name: str = None) -> str
scratchpad_get_lines(pad_name: str, section_name: str, start: int = None, end: int = None) -> str

# Writing
scratchpad_write(pad_name: str, section_name: str, content: str, mode: str = "replace") -> str

# Line-level editing
scratchpad_insert_lines(pad_name: str, section_name: str, line_number: int, content: str) -> str
scratchpad_delete_lines(pad_name: str, section_name: str, start_line: int, end_line: int = None) -> str
scratchpad_replace_lines(pad_name: str, section_name: str, start_line: int, end_line: int, content: str) -> str

# Advanced
scratchpad_edit(pad_name: str, section_name: str, old_text: str, new_text: str) -> str
scratchpad_delete(pad_name: str, section_name: str) -> str
scratchpad_merge(pad_name: str, section_names: list[str], new_section_name: str) -> str
```

## Implementation Status

✅ ScratchpadManager class with full CRUD operations  
✅ Line-level editing (insert/delete/replace lines)  
✅ 11 scratchpad tool functions  
✅ Integrated into TOOL_FUNCTIONS map  
✅ Updated TOOL_DEFINITIONS with full documentation  
✅ Updated agent personas (Orchestrator, Writer, Tool Agent)  
✅ Initialized in run_agentic_workflow()  
✅ Session state management  

## Next Steps

1. **Test workflow** with a complex multi-step query
2. **Monitor agent usage** of scratchpads (are they using line-level ops?)
3. **Add async execution** for true parallel agent work
4. **Visualize scratchpads** in Streamlit UI (show all pads side-by-side)
5. **Add versioning** (track changes to OUTPUT pad over time)
6. **Export capabilities** (download OUTPUT pad as markdown/PDF)
