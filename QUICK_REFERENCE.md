# 🚀 Scratchpad System - Quick Reference

## Agent Tools

### Read Operations
```python
scratchpad_summary()                           # Overview of all pads
scratchpad_list("output")                      # List sections in pad
scratchpad_read("output", "intro")             # Read a section
scratchpad_get_lines("output", "intro", 1, 10) # View lines 1-10
```

### Write Operations
```python
# Replace entire section
scratchpad_write("output", "intro", content, mode="replace")

# Append to section
scratchpad_write("output", "intro", content, mode="append")

# Prepend to section
scratchpad_write("output", "intro", content, mode="prepend")
```

### Line-Level Editing
```python
# Insert at specific line (1-indexed)
scratchpad_insert_lines("output", "intro", 5, "New line")

# Replace line range
scratchpad_replace_lines("output", "intro", 3, 5, "Replacement")

# Delete line range
scratchpad_delete_lines("output", "intro", 10, 12)
```

### Advanced
```python
# View version history with diffs
scratchpad_history("output", "intro", limit=10)

# Find/replace text
scratchpad_edit("output", "intro", "old text", "new text")

# Delete entire section
scratchpad_delete("output", "intro")

# Merge sections
scratchpad_merge("output", ["intro", "summary"], "combined")
```

## Available Pads

| Pad | Purpose | Example Usage |
|-----|---------|---------------|
| `output` | Final answer | `scratchpad_write("output", "intro", "# Report")` |
| `research` | Search results | `scratchpad_write("research", "findings", data, mode="append")` |
| `tables` | Formatted tables | `scratchpad_write("tables", "metrics", table_md)` |
| `plots` | Plot data | `scratchpad_write("plots", "chart1", plot_spec)` |
| `outline` | Structure/plan | `scratchpad_write("outline", "plan", "1. Intro\n2. Body")` |
| `data` | Raw JSON/lists | `scratchpad_write("data", "results", json_str)` |
| `log` | Agent actions | `scratchpad_write("log", "actions", "Step 1: ...")` |

## Diff Output Example

```diff
✅ Wrote to 'output.intro' (mode: append)

@@ -1,3 +1,4 @@
  # Report
  
  Background info.
+ Key finding: 42% improvement.
```

## Common Patterns

### Build Document Section by Section
```python
# Loop 1: Title
scratchpad_write("output", "title", "# F-35 Report")

# Loop 2: Executive Summary
scratchpad_write("output", "exec_summary", "## Summary\n\n...")

# Loop 3: Background
scratchpad_write("output", "background", "## Background\n\n...")

# Final: Read all
full_doc = scratchpad_read("output")  # All sections combined
```

### Collaborative Editing
```python
# Writer creates outline
scratchpad_write("outline", "plan", "1. Intro\n2. Analysis", agent_name="Writer")

# Tool Agent adds data
scratchpad_write("research", "data", search_results, agent_name="ToolAgent")

# Engineer adds technical details at line 5
scratchpad_insert_lines("output", "intro", 5, "Technical note...", agent_name="Engineer")

# Writer refines Engineer's addition
scratchpad_replace_lines("output", "intro", 5, 5, "Technical note (revised)...", agent_name="Writer")
```

### Check What Changed
```python
# After multiple edits, view history
scratchpad_history("output", "intro", limit=5)
```

## File Locations

```
Local:  scratchpad.db
Azure:  /tmp/scratchpad.db
```

## Troubleshooting

### Check if scratchpad is initialized
```python
if "scratchpad_manager" in st.session_state:
    print("✅ Initialized")
else:
    print("❌ Not initialized")
```

### Check database path
```python
from app_mcp import is_running_on_azure, get_scratchpad_db_path
print(f"Azure: {is_running_on_azure()}")
print(f"DB Path: {get_scratchpad_db_path()}")
```

### View all sections across all pads
```python
print(scratchpad_summary())
```

## Best Practices

1. **Use descriptive section names**: `"findings_q1_2024"` not `"data1"`
2. **Keep sections focused**: One topic per section
3. **Use line-level ops for small changes**: Don't rewrite entire sections
4. **Track agent names**: Pass `agent_name` parameter for accountability
5. **Check history when debugging**: Use `scratchpad_history()` to see what changed

## Azure Deployment

The code **automatically detects** Azure and uses `/tmp`:

```python
# No changes needed! Code handles it automatically:
if is_running_on_azure():
    db_path = "/tmp/scratchpad.db"
else:
    db_path = "scratchpad.db"
```

## Documentation

- `SCRATCHPAD_SYSTEM.md` - Full system documentation
- `ARCHITECTURE_DIAGRAM.md` - Visual workflow
- `SQLITE_SCRATCHPAD_SUMMARY.md` - Database details
- `AZURE_DEPLOYMENT_CHECKLIST.md` - Azure considerations
- `IMPLEMENTATION_COMPLETE.md` - What was built
- `QUICK_REFERENCE.md` - This file

## Need Help?

Check logs:
```bash
# Local
tail -f app.log

# Azure
az webapp log tail --name <app> --resource-group <rg>
```
