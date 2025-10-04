# Multi-Scratchpad Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER QUESTION                                │
│              "Analyze F-35 project performance metrics"              │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATOR                                   │
│  • Reviews LOG pad                                                   │
│  • Calls scratchpad_summary() to see all pad states                 │
│  • Delegates tasks to specialist agents                             │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
        ┌───────────────────┐     ┌───────────────────┐
        │   TOOL AGENT      │     │   WRITER          │
        │                   │     │                   │
        │ • Search KB       │     │ • Create outline  │
        │ • Create tables   │     │ • Build output    │
        │ • Format data     │     │ • Edit content    │
        └───────────────────┘     └───────────────────┘
                 │                         │
                 │                         │
                 ▼                         ▼

┌──────────────────────────── SCRATCHPAD MANAGER ────────────────────────────┐
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   OUTLINE   │  │  RESEARCH   │  │   TABLES    │  │   PLOTS     │     │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤  ├─────────────┤     │
│  │ • plan      │  │ • findings  │  │ • metrics   │  │ • perf_plot │     │
│  │ • structure │  │ • kb_results│  │ • comparison│  │ • timeline  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                       │
│  │   OUTPUT    │  │    DATA     │  │     LOG     │                       │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤                       │
│  │ • intro     │  │ • raw_json  │  │ • user_goal │                       │
│  │ • analysis  │  │ • metrics   │  │ • actions   │                       │
│  │ • results   │  │ • stats     │  │ • decisions │                       │
│  │ • conclusion│  │             │  │             │                       │
│  └─────────────┘  └─────────────┘  └─────────────┘                       │
│                                                                             │
│  Line-Level Operations:                                                    │
│  • scratchpad_get_lines(pad, section, start, end)                         │
│  • scratchpad_insert_lines(pad, section, line_num, content)               │
│  • scratchpad_replace_lines(pad, section, start, end, content)            │
│  • scratchpad_delete_lines(pad, section, start, end)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

AGENT WORKFLOW EXAMPLE (Parallel Operations):

Loop 1:  Writer → OUTLINE pad
         ├─ scratchpad_write("outline", "structure", "1. Intro\n2. Analysis...")

Loop 2:  Tool Agent → RESEARCH pad
         ├─ search_knowledge_base(["F-35", "performance"])
         └─ scratchpad_write("research", "kb_results", search_output, "append")

Loop 3:  Tool Agent → TABLES pad
         ├─ to_markdown_table(parsed_metrics)
         └─ scratchpad_write("tables", "metrics", table_md)

Loop 4:  Writer → OUTPUT pad (reads RESEARCH)
         ├─ research = scratchpad_read("research", "kb_results")
         ├─ scratchpad_write("output", "intro", "# F-35 Analysis\n\nOverview...")
         └─ scratchpad_insert_lines("output", "intro", -1, "Key findings: ...")

Loop 5:  Engineer → OUTPUT pad (adds technical details)
         ├─ scratchpad_get_lines("output", "intro")  # Review current state
         └─ scratchpad_insert_lines("output", "intro", 5, "Technical: 42% gain")

Loop 6:  Writer → OUTPUT pad (integrate table)
         ├─ table = scratchpad_read("tables", "metrics")
         └─ scratchpad_write("output", "results", f"## Results\n\n{table}")

Loop 7:  Writer → OUTPUT pad (refine)
         ├─ scratchpad_get_lines("output", "intro", 1, 10)
         └─ scratchpad_replace_lines("output", "intro", 5, 5, "Technical: 42.3% gain")

Loop 8:  Validator → OUTPUT pad (check)
         ├─ final = scratchpad_read("output")
         └─ return {"response": "VALIDATION_PASSED"}

Loop 9:  FINISH → Return OUTPUT pad to user

BENEFITS OF THIS ARCHITECTURE:

✅ Parallel Work: Tool Agent works on RESEARCH while Writer builds OUTLINE
✅ Efficient Edits: Only change specific lines, not entire documents
✅ Clear Separation: Research ≠ Final Output ≠ Raw Data ≠ Tables
✅ Incremental Building: Add to OUTPUT piece by piece as info arrives
✅ Collaboration: Multiple agents can read/write different sections
✅ Versioning: Each pad tracks modifications independently
✅ Reusability: Tables/plots created once, used in multiple places
