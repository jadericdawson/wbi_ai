# Azure App Service Deployment - Scratchpad Database Considerations

## ⚠️ Critical Azure Issues to Address

### 1. **Ephemeral File System**
Azure App Service containers have **ephemeral (temporary) file systems**:
- Files written to disk are **lost on restart**
- Container restarts happen during:
  - App restarts
  - Scaling operations
  - Platform updates
  - Deployment slots swaps

**Impact on scratchpad.db:**
```python
# This will create scratchpad.db locally:
manager = ScratchpadManager(db_path="scratchpad.db")
# ❌ BUT: File will be lost when container restarts!
```

### 2. **Multiple Instance Problem**
If your Azure App scales to multiple instances:
- Each instance has its **own file system**
- Instance A's `scratchpad.db` ≠ Instance B's `scratchpad.db`
- Users hitting different instances see different scratchpads
- **No synchronization between instances**

### 3. **Write Permissions**
Azure App Service containers may have restricted write permissions in certain directories.

## ✅ Solutions for Azure

### **Option 1: Azure Blob Storage** (Recommended)
Store `scratchpad.db` in Azure Blob Storage with file locking.

**Pros:**
- Persistent across restarts
- Shared across all instances
- Already using Cosmos DB, so Azure-native storage is familiar

**Cons:**
- SQLite doesn't support concurrent writes well on network storage
- Need to implement file locking/retry logic

**Implementation:**
```python
from azure.storage.blob import BlobServiceClient
import tempfile
import os

class AzureScratchpadManager(ScratchpadManager):
    def __init__(self, container_name="scratchpads", session_id=None):
        self.blob_client = BlobServiceClient(...)
        self.container = container_name
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Download DB from blob to temp location
        self.local_db_path = f"/tmp/scratchpad_{self.session_id}.db"
        self._download_from_blob()
        
        # Initialize with local path
        super().__init__(db_path=self.local_db_path, session_id=self.session_id)
    
    def _download_from_blob(self):
        """Download DB from Azure Blob to local temp"""
        blob_name = f"scratchpad_{self.session_id}.db"
        try:
            blob_client = self.blob_client.get_blob_client(
                container=self.container, 
                blob=blob_name
            )
            with open(self.local_db_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
        except ResourceNotFoundError:
            # DB doesn't exist yet, will be created
            pass
    
    def _upload_to_blob(self):
        """Upload local DB to Azure Blob"""
        blob_name = f"scratchpad_{self.session_id}.db"
        blob_client = self.blob_client.get_blob_client(
            container=self.container,
            blob=blob_name
        )
        with open(self.local_db_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
    
    def write_section(self, *args, **kwargs):
        """Override to sync with blob after write"""
        result = super().write_section(*args, **kwargs)
        self._upload_to_blob()  # Sync to blob after every write
        return result
```

### **Option 2: Azure Cosmos DB** (Best for Multi-Instance)
Replace SQLite with Cosmos DB (NoSQL).

**Pros:**
- Native Azure service (already using it!)
- Multi-instance safe
- No file system issues
- Scales automatically
- Built-in versioning/change feed

**Cons:**
- Need to rewrite ScratchpadManager
- Different query patterns than SQLite

**Implementation:**
```python
class CosmosDBScratchpadManager:
    def __init__(self, cosmos_client, database_name="scratchpads"):
        self.client = cosmos_client
        self.database = self.client.get_database_client(database_name)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collections: pads, sections, version_history
        self.pads_container = self.database.get_container_client("pads")
        self.sections_container = self.database.get_container_client("sections")
        self.history_container = self.database.get_container_client("version_history")
    
    def write_section(self, pad_name, section_name, content, mode="replace", agent_name="agent"):
        # Read existing
        try:
            section = self.sections_container.read_item(
                item=f"{self.session_id}_{pad_name}_{section_name}",
                partition_key=self.session_id
            )
            old_content = section["content"]
        except:
            old_content = ""
        
        # Apply mode
        if mode == "append":
            new_content = old_content + "\n" + content if old_content else content
        else:
            new_content = content
        
        # Generate diff
        diff = self._generate_diff(old_content, new_content)
        
        # Save section
        self.sections_container.upsert_item({
            "id": f"{self.session_id}_{pad_name}_{section_name}",
            "session_id": self.session_id,
            "pad_name": pad_name,
            "section_name": section_name,
            "content": new_content,
            "last_modified": datetime.utcnow().isoformat()
        })
        
        # Save version history
        self.history_container.create_item({
            "id": str(uuid.uuid4()),
            "session_id": self.session_id,
            "pad_name": pad_name,
            "section_name": section_name,
            "operation": f"write_{mode}",
            "diff": diff,
            "agent_name": agent_name,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Format diff for display
        diff_display = self._format_diff_for_display(diff)
        return f"✅ Wrote to '{pad_name}.{section_name}' (mode: {mode})\n\n{diff_display}"
```

### **Option 3: /tmp Directory** (Development Only)
Use `/tmp` for temporary storage during development.

**Pros:**
- Simple
- No code changes needed

**Cons:**
- Still ephemeral (lost on restart)
- Not shared across instances
- Only for testing

**Implementation:**
```python
# In app_mcp.py
if "scratchpad_manager" not in st.session_state:
    # Use /tmp for Azure ephemeral storage
    db_path = "/tmp/scratchpad.db" if os.getenv("WEBSITE_INSTANCE_ID") else "scratchpad.db"
    st.session_state.scratchpad_manager = ScratchpadManager(db_path=db_path)
```

### **Option 4: Session-Based (Streamlit Session State)**
Keep scratchpads in memory (Streamlit session state) for single-user sessions.

**Pros:**
- No file system needed
- Works in any Azure configuration
- Simple implementation

**Cons:**
- Lost when user closes browser
- Not persistent across sessions
- Can't inspect with external tools

**Implementation:**
```python
# Already works! Just don't persist to disk
# Scratchpads live in st.session_state only
# No SQLite file created
```

## 🎯 Recommended Approach for Azure

### Phase 1: Quick Fix (Use /tmp)
```python
# Update app_mcp.py line ~2480
def run_agentic_workflow(user_prompt: str, log_placeholder, final_answer_placeholder, query_expander):
    if "stop_generation" not in st.session_state:
        st.session_state.stop_generation = False

    # Initialize scratchpad manager
    if "scratchpad_manager" not in st.session_state:
        # Detect if running on Azure
        is_azure = os.getenv("WEBSITE_INSTANCE_ID") is not None
        
        if is_azure:
            # Use /tmp on Azure (ephemeral but works)
            db_path = "/tmp/scratchpad.db"
        else:
            # Use local directory for development
            db_path = "scratchpad.db"
        
        st.session_state.scratchpad_manager = ScratchpadManager(
            db_path=db_path,
            session_id=st.session_state.get("user_id", "unknown")
        )
```

### Phase 2: Production (Use Cosmos DB)
Replace SQLite with Cosmos DB for multi-instance safety:

1. Create new containers in existing Cosmos DB:
   - `scratchpad_pads`
   - `scratchpad_sections`
   - `scratchpad_versions`

2. Rewrite `ScratchpadManager` to use Cosmos DB client

3. Use session_id = user_id for per-user scratchpads

4. Version history stored in Cosmos DB with TTL for auto-cleanup

## 🔧 Immediate Action Items

### 1. Add Azure Detection
```python
import os

def is_running_on_azure():
    """Detect if running on Azure App Service"""
    return os.getenv("WEBSITE_INSTANCE_ID") is not None

def get_scratchpad_db_path():
    """Get appropriate DB path for environment"""
    if is_running_on_azure():
        return "/tmp/scratchpad.db"  # Ephemeral but works
    else:
        return "scratchpad.db"  # Local development
```

### 2. Update ScratchpadManager Initialization
```python
# In run_agentic_workflow()
if "scratchpad_manager" not in st.session_state:
    db_path = get_scratchpad_db_path()
    session_id = st.session_state.get("user_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    st.session_state.scratchpad_manager = ScratchpadManager(
        db_path=db_path,
        session_id=session_id
    )
```

### 3. Add Logging
```python
# At top of app_mcp.py
logger.info(f"Running on Azure: {is_running_on_azure()}")
logger.info(f"Scratchpad DB path: {get_scratchpad_db_path()}")
```

### 4. Test on Azure
Deploy and check:
```bash
# SSH into Azure App Service container
az webapp ssh --name <app-name> --resource-group <rg-name>

# Check if DB was created
ls -lh /tmp/scratchpad*.db

# Check logs
tail -f /home/LogFiles/Application/appinsights-*.log
```

## 📋 Testing Checklist

- [ ] App starts without errors on Azure
- [ ] `/tmp/scratchpad.db` is created
- [ ] Agents can write to scratchpads
- [ ] Diffs are displayed correctly
- [ ] Version history is queryable
- [ ] Scratchpad content persists within same session
- [ ] Handle gracefully when DB is lost on restart
- [ ] Multiple users get separate session_ids

## 🚨 Known Limitations (with /tmp solution)

1. **Data lost on restart**: Container restart = lose all scratchpad data
2. **Not shared across instances**: If scaled to 2+ instances, each has own `/tmp`
3. **No backup**: Can't recover scratchpad data if lost
4. **Storage limits**: `/tmp` has size limits (~2GB typically)

## 🎯 Long-Term Solution

**Migrate to Cosmos DB** for production:
- Persistent across restarts
- Shared across all instances
- Automatic backup/recovery
- Scales with your app
- Integrates with existing Cosmos DB setup

Would you like me to implement the Cosmos DB version?

## Environment Variables to Check

```bash
# On Azure App Service, these are set:
WEBSITE_INSTANCE_ID=<instance-id>
WEBSITE_SITE_NAME=<app-name>
HOME=/home
TMPDIR=/tmp
```

## Quick Test Script for Azure

```python
# test_azure_scratchpad.py
import os
import sqlite3

print("=" * 80)
print("AZURE SCRATCHPAD TEST")
print("=" * 80)

print(f"\n1. Environment:")
print(f"   WEBSITE_INSTANCE_ID: {os.getenv('WEBSITE_INSTANCE_ID', 'NOT SET (local)')}")
print(f"   HOME: {os.getenv('HOME', 'NOT SET')}")
print(f"   TMPDIR: {os.getenv('TMPDIR', 'NOT SET')}")

print(f"\n2. Writable Locations:")
for path in ["/tmp", "/home", "."]:
    try:
        test_file = f"{path}/test_write.txt"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        print(f"   ✅ {path} is writable")
    except:
        print(f"   ❌ {path} is NOT writable")

print(f"\n3. SQLite Test:")
try:
    db_path = "/tmp/test_sqlite.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE test (id INTEGER)")
    cursor.execute("INSERT INTO test VALUES (1)")
    conn.commit()
    conn.close()
    os.remove(db_path)
    print(f"   ✅ SQLite works in /tmp")
except Exception as e:
    print(f"   ❌ SQLite failed: {e}")

print("\n" + "=" * 80)
```

Upload this to Azure and run via:
```bash
az webapp ssh --name <app-name> --resource-group <rg-name>
python3 test_azure_scratchpad.py
```
