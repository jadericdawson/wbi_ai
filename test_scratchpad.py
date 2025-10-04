#!/usr/bin/env python3
"""Test script to demonstrate scratchpad system with SQLite and diffs"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Mock streamlit session_state
class MockSessionState:
    def __init__(self):
        self._state = {}
    
    def get(self, key, default=None):
        return self._state.get(key, default)
    
    def __setitem__(self, key, value):
        self._state[key] = value
    
    def __getitem__(self, key):
        return self._state[key]
    
    def __contains__(self, key):
        return key in self._state

# Create mock st module
class MockSt:
    def __init__(self):
        self.session_state = MockSessionState()

sys.modules['streamlit'] = MockSt()
import streamlit as st

# Now import the scratchpad manager
import sqlite3
from datetime import datetime
import difflib

# Copy just the ScratchpadManager class for testing
exec(open('/home/jadericdawson/Documents/AI/AGENTS/app_mcp.py').read().split('class ScratchpadManager:')[1].split('\n\nTOOL_DEFINITIONS')[0])

# Test the scratchpad system
print("=" * 80)
print("SCRATCHPAD SYSTEM TEST - SQLite + Diff Tracking")
print("=" * 80)

# Initialize manager
manager = ScratchpadManager(db_path="test_scratchpad.db")
print(f"\n✅ Initialized scratchpad database: test_scratchpad.db")
print(f"   Session ID: {manager.session_id}\n")

# Test 1: Write to output pad
print("-" * 80)
print("TEST 1: Write initial content to OUTPUT pad")
print("-" * 80)
result = manager.write_section("output", "introduction", 
"""# Analysis Report

This report examines project performance metrics.""", agent_name="Writer")
print(result)

# Test 2: Append to same section
print("\n" + "-" * 80)
print("TEST 2: Append additional content")
print("-" * 80)
result = manager.write_section("output", "introduction", 
"Key findings indicate significant improvements.", mode="append", agent_name="Engineer")
print(result)

# Test 3: Write to research pad
print("\n" + "-" * 80)
print("TEST 3: Tool Agent writes search results to RESEARCH pad")
print("-" * 80)
result = manager.write_section("research", "kb_results",
"""Found 10 documents related to F-35 project:
- Document 1: Performance metrics Q1 2024
- Document 2: Budget analysis
- Document 3: Timeline updates""", agent_name="ToolAgent")
print(result)

# Test 4: Check summary
print("\n" + "-" * 80)
print("TEST 4: Get scratchpad summary")
print("-" * 80)
print(manager.get_all_pads_summary())

# Test 5: View version history
print("\n" + "-" * 80)
print("TEST 5: View version history for output.introduction")
print("-" * 80)
history = manager.get_version_history("output", "introduction")
for i, v in enumerate(history, 1):
    print(f"\nVersion {i}:")
    print(f"  Operation: {v['operation']}")
    print(f"  Agent: {v['agent']}")
    print(f"  Timestamp: {v['timestamp']}")
    print(f"  Diff:\n{v['diff']}")

# Test 6: Read final content
print("\n" + "-" * 80)
print("TEST 6: Read final OUTPUT pad content")
print("-" * 80)
content = manager.read_section("output", "introduction")
print(content)

print("\n" + "=" * 80)
print("DATABASE LOCATION: test_scratchpad.db")
print("Use: sqlite3 test_scratchpad.db")
print("Tables: pads, sections, version_history")
print("=" * 80)
