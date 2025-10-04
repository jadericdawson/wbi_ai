# 🔧 JSON Response Format Fix

## Problem

Azure OpenAI API requires that when using `response_format={"type": "json_object"}`, the system message **must contain the word "json"** explicitly.

### Error Message
```
'messages' must contain the word 'json' in some form, to use 'response_format' of type 'json_object'
```

## Solution Applied

Updated all agent personas to explicitly mention JSON in their system messages:

### ✅ Updated Agents

1. **Writer** - Line 2340
   - Added: "You MUST respond in JSON format"
   - Added section 6: "**RESPONSE FORMAT**: You MUST respond with a valid JSON object"
   - Added section 7: "NEVER respond with plain text. Always use JSON format"

2. **Engineer** - Line 2333  
   - Added: "You MUST respond in JSON format"
   - Added: "You MUST respond with a valid JSON object"
   - Added: "NEVER respond with plain text. Always use JSON format"

3. **Validator** - Line 2364
   - Added: "You MUST respond in JSON format"
   - Added: "NEVER respond with plain text. Always use JSON format"

4. **Supervisor** - Line 2380
   - Added: "You MUST respond in JSON format"  
   - Added: "NEVER respond with plain text. Always use JSON format with {\"response\": \"...\"}"

5. **Query Refiner** - Line 2307
   - Added: "You MUST respond in JSON format"
   - Added: "NEVER respond with plain text. Always use JSON format"

### Pattern Used

Every agent persona now includes:
```python
"""You are a [role description]. You MUST respond in JSON format.

CRITICAL INSTRUCTIONS:
...
[instructions specific to agent]
...
X. NEVER respond with plain text. Always use JSON format."""
```

## Testing

After this fix, agents will:
1. ✅ Always include "json" in their system prompt
2. ✅ Satisfy Azure OpenAI API requirements
3. ✅ Understand they must respond in JSON format
4. ✅ Not return empty responses due to API errors

## Files Modified

- `app_mcp.py` - Updated 5 agent personas in `AGENT_PERSONAS` dict

## What This Fixes

Your specific error:
```
Loop 1: Error executing task for Writer: Error code: 400 - 
{'error': {'message': "'messages' must contain the word 'json' in some form..."}}
```

Now the Writer (and all other agents) will work correctly with Azure OpenAI's JSON mode.

## Related Documentation

- Azure OpenAI JSON Mode: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/json-mode
- Agent Personas: See `AGENT_PERSONAS` dict in `app_mcp.py` (starting line ~2252)

## Next Steps

1. **Deploy** - The fix is ready
2. **Test** - Try the same F-35 manpower query
3. **Monitor** - Check that agents respond with JSON

The error should be resolved! 🎉
