# DeepLake 4+ Backup & Cloud Sync Guide

**⚠️ IMPORTANT: This guide is for DeepLake version 4+ only. Pre-v4 APIs may differ.**

## Table of Contents
1. [Weekly Backup Strategy](#weekly-backup-strategy)
2. [Cloud Sync with Versioning](#cloud-sync-with-versioning)
3. [Migration: Local to Cloud](#migration-local-to-cloud)
4. [Automated Backup Script](#automated-backup-script)
5. [Deep Dive: Components, Version Control & History](#deep-dive-components-version-control--history)
   - [Dataset Components Explained](#dataset-components-explained)
   - [Overwrite Protection Mechanisms](#overwrite-protection-mechanisms)
   - [Versions and Commits Deep Dive](#versions-and-commits-deep-dive)
   - [History Lookback & Time Travel](#history-lookback--time-travel)

---

## Weekly Backup Strategy

### Method 1: Using `deeplake.copy()` (Recommended)

The `deeplake.copy()` method preserves **all versions, metadata, and history** when creating backups.

```python
import deeplake
from datetime import datetime
import os

def weekly_backup(source_path, backup_base_dir):
    """
    Create a weekly backup of a DeepLake dataset.
    
    Args:
        source_path: Path to your local DeepLake dataset (e.g., "./my_dataset")
        backup_base_dir: Base directory for backups (e.g., "./backups")
    """
    # Create timestamped backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_base_dir, f"backup_{timestamp}")
    
    # Ensure backup directory exists
    os.makedirs(backup_base_dir, exist_ok=True)
    
    print(f"Starting backup from {source_path} to {backup_path}...")
    
    # Copy dataset with all versions preserved
    deeplake.copy(
        source=source_path,
        dest=backup_path,
        overwrite=False,  # Set to True if you want to overwrite existing backups
        multiprocessing=True,  # Speed up large datasets
        verbose=True
    )
    
    print(f"✓ Backup completed: {backup_path}")
    return backup_path

# Usage
if __name__ == "__main__":
    local_dataset = "./my_local_dataset"
    backup_dir = "./weekly_backups"
    
    weekly_backup(local_dataset, backup_dir)
```

### Method 2: Version-Based Backup (Before Major Changes)

Before making significant changes, commit and create a versioned backup:

```python
import deeplake
from datetime import datetime

def create_versioned_backup(dataset_path, backup_location):
    """
    Create a backup with explicit version tagging.
    """
    # Open dataset
    ds = deeplake.dataset(dataset_path, write=True)
    
    # Commit current state with descriptive message
    commit_id = ds.commit(f"Weekly backup - {datetime.now().isoformat()}")
    
    # Create a tag for easy reference
    ds.tag = f"backup_{datetime.now().strftime('%Y%m%d')}"
    
    # Copy to backup location
    deeplake.copy(dataset_path, backup_location, overwrite=False)
    
    print(f"✓ Versioned backup created: {backup_location}")
    print(f"  Commit ID: {commit_id}")
    print(f"  Tag: {ds.tag}")
    
    return commit_id

# Usage
create_versioned_backup("./my_local_dataset", "./backups/weekly_backup")
```

### Best Practices for Weekly Backups

1. **Automate with Cron/Scheduler**: Use system cron jobs or task schedulers
2. **Retention Policy**: Keep last 4-8 weekly backups, delete older ones
3. **Verify Backups**: Periodically test restoring from backups
4. **Separate Storage**: Store backups on different drives/cloud storage
5. **Monitor Disk Space**: Large datasets can fill up backup storage quickly

---

## Cloud Sync with Versioning

### Setting Up Cloud Storage

First, configure your cloud storage credentials:

```python
import deeplake
import os

# For Activeloop Hub
deeplake.login(token="your_activeloop_token")

# For AWS S3
os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'

# For Google Cloud Storage
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/service-account.json'

# For Azure Blob Storage
os.environ['AZURE_STORAGE_ACCOUNT_NAME'] = 'your_account_name'
os.environ['AZURE_STORAGE_ACCOUNT_KEY'] = 'your_account_key'
```

### Pushing Local Changes to Cloud

**Step-by-step process to sync local changes to cloud with versioning:**

```python
import deeplake
from datetime import datetime

def push_to_cloud(local_path, cloud_path, commit_message=None):
    """
    Push local dataset changes to cloud storage with versioning.
    
    Args:
        local_path: Local dataset path (e.g., "./my_local_dataset")
        cloud_path: Cloud storage path (e.g., "hub://username/my_dataset" or "s3://bucket/dataset")
        commit_message: Optional commit message
    """
    # Open local dataset in write mode
    ds = deeplake.dataset(local_path, write=True)
    
    # Check if there are uncommitted changes
    if ds.has_changes:
        # Commit changes with descriptive message
        if commit_message is None:
            commit_message = f"Sync to cloud - {datetime.now().isoformat()}"
        
        commit_id = ds.commit(commit_message)
        print(f"✓ Committed changes: {commit_id}")
        print(f"  Message: {commit_message}")
    else:
        print("No uncommitted changes to push.")
        return
    
    # Push to cloud (this syncs all committed versions)
    print(f"Pushing to {cloud_path}...")
    ds.push(
        destination=cloud_path,
        num_workers=None,  # Auto-detect optimal workers
        progress_bar=True,
        verbose=True
    )
    
    print(f"✓ Successfully pushed to {cloud_path}")
    
    # Verify by checking cloud dataset
    cloud_ds = deeplake.dataset(cloud_path)
    print(f"Cloud dataset has {len(cloud_ds.log())} commits")

# Usage
push_to_cloud(
    local_path="./my_local_dataset",
    cloud_path="hub://username/my_dataset",
    commit_message="Weekly sync - new data added"
)
```

### Pulling Cloud Changes to Local

**Sync cloud dataset to local (useful for collaboration):**

```python
import deeplake

def pull_from_cloud(cloud_path, local_path, overwrite=False):
    """
    Pull latest version from cloud to local.
    
    Args:
        cloud_path: Cloud storage path
        local_path: Local destination path
        overwrite: Whether to overwrite existing local dataset
    """
    print(f"Pulling from {cloud_path} to {local_path}...")
    
    deeplake.pull(
        source=cloud_path,
        dest=local_path,
        overwrite=overwrite,  # Set True to overwrite local changes
        num_workers=None,  # Auto-detect optimal workers
        progress_bar=True,
        verbose=True
    )
    
    # Verify local dataset
    local_ds = deeplake.dataset(local_path)
    print(f"✓ Local dataset synced")
    print(f"  Total commits: {len(local_ds.log())}")
    
    return local_ds

# Usage
pull_from_cloud(
    cloud_path="hub://username/my_dataset",
    local_path="./my_local_dataset",
    overwrite=False  # Set True if you want to discard local changes
)
```

### Bidirectional Sync Strategy

**For maintaining sync between local and cloud:**

```python
import deeplake
from datetime import datetime

def sync_local_to_cloud(local_path, cloud_path):
    """
    Smart sync: Pull cloud changes first, then push local changes.
    This prevents conflicts and ensures both sides are up-to-date.
    """
    print("=== Starting bidirectional sync ===\n")
    
    # Step 1: Check if cloud dataset exists
    try:
        cloud_ds = deeplake.dataset(cloud_path)
        cloud_commits = len(cloud_ds.log())
        print(f"Cloud dataset found with {cloud_commits} commits")
        
        # Step 2: Pull latest from cloud (if local exists)
        try:
            local_ds = deeplake.dataset(local_path)
            local_commits = len(local_ds.log())
            print(f"Local dataset found with {local_commits} commits")
            
            if cloud_commits > local_commits:
                print("\nCloud has newer commits. Pulling...")
                deeplake.pull(cloud_path, local_path, overwrite=False)
        except:
            print("Local dataset doesn't exist. Pulling from cloud...")
            deeplake.pull(cloud_path, local_path, overwrite=False)
    
    except:
        print("Cloud dataset doesn't exist. Will create it during push.")
    
    # Step 3: Open local dataset and check for changes
    local_ds = deeplake.dataset(local_path, write=True)
    
    if local_ds.has_changes:
        # Step 4: Commit local changes
        commit_id = local_ds.commit(f"Sync - {datetime.now().isoformat()}")
        print(f"\n✓ Committed local changes: {commit_id}")
        
        # Step 5: Push to cloud
        print("Pushing to cloud...")
        local_ds.push(cloud_path, progress_bar=True, verbose=True)
        print("✓ Sync complete!")
    else:
        print("No local changes to sync.")

# Usage
sync_local_to_cloud(
    local_path="./my_local_dataset",
    cloud_path="hub://username/my_dataset"
)
```

---

## Migration: Local to Cloud

### Complete Migration with Full Version History

**This is EXTREMELY IMPORTANT - preserving all versions during migration:**

```python
import deeplake
from datetime import datetime

def migrate_local_to_cloud(local_path, cloud_path, verify=True):
    """
    Migrate local dataset to cloud storage with FULL version history preserved.
    
    This is a one-time migration that copies everything including:
    - All commits and version history
    - All branches and tags
    - All metadata
    - All data chunks
    
    Args:
        local_path: Local dataset path
        cloud_path: Cloud destination path (e.g., "hub://username/dataset" or "s3://bucket/dataset")
        verify: Whether to verify the migration by comparing versions
    """
    print("=" * 60)
    print("DEEP LAKE MIGRATION: Local to Cloud")
    print("=" * 60)
    
    # Step 1: Verify local dataset
    print(f"\n[1/5] Verifying local dataset: {local_path}")
    local_ds = deeplake.dataset(local_path)
    local_commits = local_ds.log()
    print(f"   ✓ Local dataset verified")
    print(f"   - Total commits: {len(local_commits)}")
    print(f"   - Branches: {list(local_ds.branches) if hasattr(local_ds, 'branches') else 'N/A'}")
    
    # Step 2: Commit any uncommitted changes before migration
    print(f"\n[2/5] Checking for uncommitted changes...")
    if local_ds.has_changes:
        commit_id = local_ds.commit(f"Pre-migration commit - {datetime.now().isoformat()}")
        print(f"   ✓ Committed uncommitted changes: {commit_id}")
    else:
        print(f"   ✓ No uncommitted changes")
    
    # Step 3: Copy to cloud (preserves ALL versions)
    print(f"\n[3/5] Copying dataset to cloud: {cloud_path}")
    print(f"   This may take a while for large datasets...")
    
    deeplake.copy(
        source=local_path,
        dest=cloud_path,
        overwrite=False,  # Set True if cloud dataset already exists and you want to replace it
        multiprocessing=True,  # Use multiple processes for faster copy
        verbose=True
    )
    
    print(f"   ✓ Dataset copied to cloud")
    
    # Step 4: Verify cloud dataset
    if verify:
        print(f"\n[4/5] Verifying cloud dataset...")
        cloud_ds = deeplake.dataset(cloud_path)
        cloud_commits = cloud_ds.log()
        
        print(f"   ✓ Cloud dataset verified")
        print(f"   - Total commits: {len(cloud_commits)}")
        
        # Compare commit counts
        if len(cloud_commits) == len(local_commits):
            print(f"   ✓ Version history preserved: {len(cloud_commits)} commits")
        else:
            print(f"   ⚠ WARNING: Commit count mismatch!")
            print(f"      Local: {len(local_commits)}, Cloud: {len(cloud_commits)}")
    
    # Step 5: Test access
    print(f"\n[5/5] Testing cloud dataset access...")
    test_ds = deeplake.dataset(cloud_path)
    sample_count = len(test_ds)
    print(f"   ✓ Cloud dataset accessible")
    print(f"   - Samples: {sample_count}")
    
    print("\n" + "=" * 60)
    print("✓ MIGRATION COMPLETE!")
    print("=" * 60)
    print(f"\nYour dataset is now available at: {cloud_path}")
    print(f"All {len(local_commits)} versions have been preserved.")
    
    return cloud_path

# Usage
migrate_local_to_cloud(
    local_path="./my_local_dataset",
    cloud_path="hub://username/my_dataset",  # Or "s3://my-bucket/my_dataset"
    verify=True
)
```

### Post-Migration: Setting Up Sync

After migration, set up regular sync:

```python
# After initial migration, use push/pull for ongoing sync
# Local remains your working copy, cloud is the backup/sync target

# Daily sync workflow:
sync_local_to_cloud(
    local_path="./my_local_dataset",
    cloud_path="hub://username/my_dataset"
)
```

---

## Automated Backup Script

**Complete automated weekly backup solution:**

```python
#!/usr/bin/env python3
"""
DeepLake Weekly Backup Script
Automates weekly backups with retention policy and cloud sync.
"""

import deeplake
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deeplake_backup.log'),
        logging.StreamHandler()
    ]
)

class DeepLakeBackupManager:
    def __init__(self, local_dataset_path, backup_dir, cloud_path=None, max_backups=8):
        """
        Initialize backup manager.
        
        Args:
            local_dataset_path: Path to local DeepLake dataset
            backup_dir: Directory for local backups
            cloud_path: Optional cloud path for sync (e.g., "hub://username/dataset")
            max_backups: Maximum number of weekly backups to keep
        """
        self.local_dataset_path = Path(local_dataset_path)
        self.backup_dir = Path(backup_dir)
        self.cloud_path = cloud_path
        self.max_backups = max_backups
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self):
        """Create a new weekly backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        logging.info(f"Creating backup: {backup_path}")
        
        try:
            deeplake.copy(
                source=str(self.local_dataset_path),
                dest=str(backup_path),
                overwrite=False,
                multiprocessing=True,
                verbose=True
            )
            logging.info(f"✓ Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logging.error(f"✗ Backup failed: {e}")
            raise
    
    def cleanup_old_backups(self):
        """Remove backups older than max_backups limit."""
        backups = sorted(self.backup_dir.glob("backup_*"), key=os.path.getmtime)
        
        if len(backups) > self.max_backups:
            to_remove = backups[:-self.max_backups]
            for backup in to_remove:
                logging.info(f"Removing old backup: {backup}")
                shutil.rmtree(backup)
    
    def sync_to_cloud(self):
        """Sync local dataset to cloud storage."""
        if not self.cloud_path:
            logging.warning("No cloud path configured. Skipping cloud sync.")
            return
        
        logging.info(f"Syncing to cloud: {self.cloud_path}")
        
        try:
            ds = deeplake.dataset(str(self.local_dataset_path), write=True)
            
            if ds.has_changes:
                commit_id = ds.commit(f"Weekly backup - {datetime.now().isoformat()}")
                logging.info(f"Committed changes: {commit_id}")
            
            ds.push(self.cloud_path, progress_bar=True, verbose=True)
            logging.info(f"✓ Synced to cloud: {self.cloud_path}")
        except Exception as e:
            logging.error(f"✗ Cloud sync failed: {e}")
            raise
    
    def run_weekly_backup(self):
        """Execute complete weekly backup workflow."""
        logging.info("=" * 60)
        logging.info("Starting weekly backup workflow")
        logging.info("=" * 60)
        
        try:
            # Step 1: Create local backup
            backup_path = self.create_backup()
            
            # Step 2: Cleanup old backups
            self.cleanup_old_backups()
            
            # Step 3: Sync to cloud (if configured)
            if self.cloud_path:
                self.sync_to_cloud()
            
            logging.info("=" * 60)
            logging.info("✓ Weekly backup workflow completed successfully")
            logging.info("=" * 60)
            
        except Exception as e:
            logging.error(f"✗ Backup workflow failed: {e}")
            raise

# Usage
if __name__ == "__main__":
    # Configure your paths
    manager = DeepLakeBackupManager(
        local_dataset_path="./my_local_dataset",
        backup_dir="./weekly_backups",
        cloud_path="hub://username/my_dataset",  # Optional
        max_backups=8  # Keep last 8 weekly backups
    )
    
    # Run weekly backup
    manager.run_weekly_backup()
```

### Setting Up Cron Job (Linux/Mac)

Add to crontab for weekly backups (every Sunday at 2 AM):

```bash
# Edit crontab
crontab -e

# Add this line (adjust path to your script)
0 2 * * 0 /usr/bin/python3 /path/to/deeplake_backup_script.py >> /path/to/backup.log 2>&1
```

### Setting Up Task Scheduler (Windows)

1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Weekly, Sunday, 2:00 AM
4. Action: Start a program
5. Program: `python.exe`
6. Arguments: `C:\path\to\deeplake_backup_script.py`

---

## Deep Dive: Components, Version Control & History

**⚠️ This section covers DeepLake 4+ version control internals, components, and history lookback.**

### Table of Contents
1. [Dataset Components Explained](#dataset-components-explained)
2. [Overwrite Protection Mechanisms](#overwrite-protection-mechanisms)
3. [Versions and Commits Deep Dive](#versions-and-commits-deep-dive)
4. [History Lookback & Time Travel](#history-lookback--time-travel)

---

### Dataset Components Explained

DeepLake datasets are composed of several key components that work together to provide efficient storage and versioning:

#### 1. **Tensors** (Data Containers)

Tensors are the fundamental data containers in DeepLake, similar to NumPy arrays or database columns. Each tensor stores a specific type of data:

```python
import deeplake
import numpy as np

# Create a dataset with multiple tensors
ds = deeplake.empty("./my_dataset")

with ds:
    # Create tensors with different types
    ds.create_tensor("images", htype="image", dtype="uint8")
    ds.create_tensor("labels", htype="class_label", dtype="int32")
    ds.create_tensor("embeddings", htype="embedding", dtype="float32")
    ds.create_tensor("metadata", htype="json")

# Add data to tensors
with ds:
    ds.images.append(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    ds.labels.append(0)
    ds.embeddings.append(np.random.rand(128).astype(np.float32))
    ds.metadata.append({"source": "camera", "timestamp": "2024-01-01"})

print(f"Dataset has {len(ds.tensors)} tensors:")
for tensor_name in ds.tensors:
    print(f"  - {tensor_name}: {ds[tensor_name].dtype}, shape={ds[tensor_name].shape}")
```

**Key Tensor Properties:**
- **htype**: Semantic type (image, video, audio, text, embedding, etc.)
- **dtype**: Data type (uint8, int32, float32, etc.)
- **Shape**: Dimensions of the tensor data
- **Chunks**: Data is stored in chunks for efficient access

#### 2. **Metadata** (Dataset Information)

Metadata includes information about the dataset itself and individual samples:

```python
# Dataset-level metadata
ds.info.update({
    "description": "My image classification dataset",
    "version": "1.0",
    "author": "Your Name",
    "created": "2024-01-01"
})

# Sample-level metadata (stored per sample)
with ds:
    ds.images[0].info.update({"source": "camera_1", "quality": "high"})
    ds.labels[0].info.update({"confidence": 0.95})

# Access metadata
print("Dataset info:", ds.info)
print("First image info:", ds.images[0].info)
```

#### 3. **Chunks** (Storage Units)

DeepLake stores data in **chunks** for efficient storage and retrieval:

```python
# Chunks are automatically managed, but you can inspect them
print(f"Tensor 'images' has {len(ds.images.chunk_engine.chunks)} chunks")

# Chunk size can be configured (affects performance)
ds = deeplake.empty("./optimized_dataset", chunk_engine="tensor_db")
with ds:
    ds.create_tensor("data", chunk_size=1024*1024)  # 1MB chunks
```

**How Chunks Work:**
- Data is divided into fixed-size chunks
- Only modified chunks are rewritten during commits
- Unchanged chunks are shared across versions (copy-on-write)
- Enables efficient versioning of large datasets

#### 4. **Version Control Tree** (DAG Structure)

DeepLake uses a **Directed Acyclic Graph (DAG)** to represent version history:

```
                    [Initial Commit]
                         |
                    [Commit 1]
                    /        \
            [Commit 2]    [Branch: experiment]
                |              |
            [Commit 3]    [Commit 4]
                |              |
            [HEAD/main]   [HEAD/experiment]
```

**Key Concepts:**
- Each node = a commit (immutable snapshot)
- Edges = parent-child relationships
- Branches = pointers to specific commits
- HEAD = current commit on a branch

---

### Overwrite Protection Mechanisms

DeepLake provides multiple mechanisms to prevent accidental data loss:

#### 1. **Read-Only Mode** (Safest for Reading)

Open datasets in read-only mode when you only need to read data:

```python
import deeplake

# Open in read-only mode (no lock acquired, no write access)
ds_readonly = deeplake.dataset("./my_dataset", read_only=True)

# Reading is allowed
data = ds_readonly.images[0].numpy()
print(f"Read {len(data)} samples")

# Writing will raise PermissionError
try:
    with ds_readonly:
        ds_readonly.images.append(new_image)
except PermissionError as e:
    print(f"✓ Protected: {e}")

ds_readonly.close()  # Always close when done
```

**When to Use Read-Only:**
- ✅ Viewing/analyzing data
- ✅ Training models (read-only access)
- ✅ Multiple processes reading simultaneously
- ✅ Preventing accidental modifications

#### 2. **File-Based Locks** (Write Protection)

DeepLake automatically creates lock files when opening datasets in write mode:

```python
# First process opens dataset (acquires lock)
ds1 = deeplake.dataset("./my_dataset", write=True)
print("Dataset opened, lock acquired")

# Second process tries to open (will wait or timeout)
try:
    ds2 = deeplake.dataset(
        "./my_dataset", 
        write=True,
        lock_timeout=5  # Wait 5 seconds for lock
    )
except TimeoutError as e:
    print(f"✓ Lock protection: {e}")

# Release lock by closing
ds1.close()
print("Lock released")

# Now second process can open
ds2 = deeplake.dataset("./my_dataset", write=True)
ds2.close()
```

**Lock Behavior:**
- **Default**: Datasets open in write mode create a lock file
- **Lock Timeout**: Set `lock_timeout` to wait for locks (seconds)
- **Lock Location**: Stored in dataset directory as `.lock` file
- **Automatic Release**: Locks released when dataset is closed

#### 3. **Version Control as Protection**

Commits create immutable snapshots, protecting against data loss:

```python
# Make changes
ds = deeplake.dataset("./my_dataset", write=True)

with ds:
    ds.images.append(new_image)
    ds.labels.append(new_label)

# Commit creates immutable snapshot
commit_id = ds.commit("Added new samples")
print(f"✓ Changes committed: {commit_id}")

# Even if you accidentally delete data, you can recover:
# ds.checkout(commit_id)  # Restore to previous state
```

**Protection Strategy:**
1. **Commit Frequently**: Create checkpoints before major changes
2. **Use Branches**: Experiment on branches, merge when ready
3. **Never Overwrite**: Use `overwrite=False` in copy operations
4. **Verify Before Overwrite**: Always check `overwrite` parameter

#### 4. **Explicit Overwrite Protection**

Protect against accidental overwrites in copy/pull operations:

```python
# Safe: Check if destination exists before copying
import os

dest_path = "./backup_dataset"
if os.path.exists(dest_path):
    response = input(f"{dest_path} exists. Overwrite? (yes/no): ")
    if response.lower() != "yes":
        print("Operation cancelled")
    else:
        deeplake.copy("./my_dataset", dest_path, overwrite=True)
else:
    deeplake.copy("./my_dataset", dest_path, overwrite=False)

# Safe: Use read-only for verification
backup_ds = deeplake.dataset(dest_path, read_only=True)
print(f"✓ Backup verified: {len(backup_ds)} samples")
backup_ds.close()
```

---

### Versions and Commits Deep Dive

#### Understanding Commits

A **commit** is an immutable snapshot of your dataset at a specific point in time:

```python
import deeplake
import numpy as np
from datetime import datetime

ds = deeplake.empty("./versioned_dataset")

with ds:
    ds.create_tensor("data", htype="generic")

# Initial commit
with ds:
    ds.data.extend([1, 2, 3, 4, 5])

commit_1 = ds.commit("Initial data: 5 samples")
print(f"Commit 1 ID: {commit_1}")
print(f"Commit 1 message: Initial data: 5 samples")
print(f"Dataset length: {len(ds)}")  # 5

# Add more data and commit
with ds:
    ds.data.extend([6, 7, 8])

commit_2 = ds.commit("Added 3 more samples")
print(f"\nCommit 2 ID: {commit_2}")
print(f"Dataset length: {len(ds)}")  # 8

# Each commit is immutable - you can't modify it
# But you can create new commits with changes
```

**Commit Properties:**
- **Unique ID**: Each commit has a unique identifier (hash)
- **Timestamp**: Automatically recorded
- **Message**: Descriptive message (required)
- **Parent**: Reference to previous commit (forms DAG)
- **Immutable**: Once created, commits cannot be modified

#### Commit Structure

```python
# View commit details
log = ds.log()

for i, commit in enumerate(log):
    print(f"\nCommit {i+1}:")
    print(f"  ID: {commit['commit_id']}")
    print(f"  Message: {commit['message']}")
    print(f"  Timestamp: {commit['timestamp']}")
    print(f"  Author: {commit.get('author', 'N/A')}")
    print(f"  Parent: {commit.get('parent', 'None (root)')}")
```

#### Working with Branches

Branches allow parallel development and experimentation:

```python
# Create a new branch for experimentation
ds.branch("experiment")
ds.checkout("experiment")
print(f"Current branch: {ds.branch()}")

# Make experimental changes
with ds:
    ds.data.extend([9, 10, 11])

experiment_commit = ds.commit("Experimental: added 3 samples")
print(f"Experiment commit: {experiment_commit}")

# Switch back to main branch
ds.checkout("main")
print(f"Back on main branch, length: {len(ds)}")  # Still 8

# View all branches
branches = ds.branches
print(f"Available branches: {branches}")

# Merge experiment branch into main (if changes are good)
# ds.merge("experiment")  # Uncomment to merge
```

**Branch Workflow:**
1. Create branch: `ds.branch("branch_name")`
2. Checkout branch: `ds.checkout("branch_name")`
3. Make changes and commit
4. Switch back: `ds.checkout("main")`
5. Merge when ready: `ds.merge("branch_name")`

#### Comparing Versions with Diff

Compare differences between commits or branches:

```python
# Compare two commits
diff = ds.diff(commit_1, commit_2)
print("Differences between commit 1 and commit 2:")
print(diff)

# Compare current state with a commit
current_diff = ds.diff(commit_1, "HEAD")
print("\nDifferences from commit 1 to current:")
print(current_diff)

# Compare branches
ds.checkout("experiment")
experiment_diff = ds.diff("main", "experiment")
print("\nDifferences between main and experiment:")
print(experiment_diff)
```

**Diff Output Format:**
- Shows added/removed/modified samples
- Indicates which tensors changed
- Provides sample-level change details
- Useful for understanding dataset evolution

#### Commit History Navigation

Navigate through commit history:

```python
# Get full commit log
log = ds.log()
print(f"Total commits: {len(log)}")

# Access specific commits
first_commit = log[0]
latest_commit = log[-1]

print(f"\nFirst commit: {first_commit['message']}")
print(f"Latest commit: {latest_commit['message']}")

# Iterate through history
print("\nCommit History:")
for i, commit in enumerate(log):
    print(f"  {i+1}. {commit['message']} ({commit['commit_id'][:8]}...)")

# Get commits by message pattern
experimental_commits = [c for c in log if "experiment" in c['message'].lower()]
print(f"\nExperimental commits: {len(experimental_commits)}")
```

---

### History Lookback & Time Travel

DeepLake's version control enables "time travel" - accessing data at any point in history:

#### 1. **Checkout Specific Commits**

Access data as it existed at a specific commit:

```python
# Current state
print(f"Current dataset length: {len(ds)}")  # 8

# Travel back to commit 1
ds.checkout(commit_1)
print(f"At commit 1, length: {len(ds)}")  # 5
print(f"Data at commit 1: {ds.data.numpy()}")

# Access specific data at that commit
image_at_commit1 = ds.images[0].numpy()  # If images exist

# Travel forward to latest
ds.checkout("HEAD")
print(f"Back to HEAD, length: {len(ds)}")  # 8
```

#### 2. **Accessing Historical Data**

Read data from any version without modifying current state:

```python
# Method 1: Checkout, read, checkout back
current_branch = ds.branch()
current_commit = ds.commit_id

# Checkout historical version
ds.checkout(commit_1)
historical_data = ds.data.numpy().copy()  # Copy to preserve

# Restore original state
ds.checkout(current_commit)

print(f"Historical data (commit 1): {historical_data}")
print(f"Current data: {ds.data.numpy()}")
```

#### 3. **Traversing Version Control Tree**

Navigate the version history tree:

```python
def traverse_history(ds, commit_id=None, depth=0, max_depth=10):
    """Recursively traverse commit history."""
    if depth > max_depth:
        return
    
    if commit_id is None:
        commit_id = ds.commit_id
    
    log = ds.log()
    commit = next((c for c in log if c['commit_id'] == commit_id), None)
    
    if commit:
        indent = "  " * depth
        print(f"{indent}Commit: {commit['message'][:50]}")
        print(f"{indent}  ID: {commit['commit_id'][:8]}...")
        print(f"{indent}  Timestamp: {commit['timestamp']}")
        
        # Checkout to see data
        ds.checkout(commit_id)
        print(f"{indent}  Samples: {len(ds)}")
        
        # Traverse parent
        if commit.get('parent'):
            traverse_history(ds, commit['parent'], depth + 1, max_depth)

# Traverse from current commit
print("Version Control Tree:")
traverse_history(ds)
```

#### 4. **Time-Based Queries**

Find commits by time range:

```python
from datetime import datetime, timedelta

# Get commits from last week
log = ds.log()
week_ago = datetime.now() - timedelta(days=7)

recent_commits = [
    c for c in log 
    if datetime.fromisoformat(c['timestamp'].replace('Z', '+00:00')) > week_ago
]

print(f"Commits in last week: {len(recent_commits)}")
for commit in recent_commits:
    print(f"  - {commit['message']} ({commit['timestamp']})")
```

#### 5. **Restoring from History**

Recover data from previous versions:

```python
def restore_to_commit(ds, target_commit_id, create_backup=True):
    """Restore dataset to a specific commit."""
    if create_backup:
        # Create backup of current state
        backup_commit = ds.commit("Backup before restore")
        print(f"✓ Backup created: {backup_commit}")
    
    # Checkout target commit
    ds.checkout(target_commit_id)
    print(f"✓ Restored to commit: {target_commit_id}")
    
    # Verify restoration
    print(f"Dataset length: {len(ds)}")
    return ds

# Restore to a previous commit
restore_to_commit(ds, commit_1, create_backup=True)
```

#### 6. **Comparing Historical States**

Compare data across different time points:

```python
def compare_versions(ds, commit1_id, commit2_id):
    """Compare dataset state at two different commits."""
    # Get state at commit 1
    ds.checkout(commit1_id)
    data1 = ds.data.numpy().copy()
    len1 = len(ds)
    
    # Get state at commit 2
    ds.checkout(commit2_id)
    data2 = ds.data.numpy().copy()
    len2 = len(ds)
    
    # Compare
    print(f"Commit 1 ({commit1_id[:8]}...): {len1} samples")
    print(f"Commit 2 ({commit2_id[:8]}...): {len2} samples")
    print(f"Difference: {len2 - len1} samples")
    
    # Show what changed
    if len1 < len2:
        new_data = data2[len1:]
        print(f"New data added: {new_data}")
    
    # Restore to HEAD
    ds.checkout("HEAD")
    
    return {
        'commit1': {'length': len1, 'data': data1},
        'commit2': {'length': len2, 'data': data2},
        'difference': len2 - len1
    }

# Compare two commits
comparison = compare_versions(ds, commit_1, commit_2)
```

#### 7. **Complete Time Travel Example**

Full example demonstrating history lookback:

```python
import deeplake
import numpy as np

# Create dataset with history
ds = deeplake.empty("./time_travel_demo")

with ds:
    ds.create_tensor("timestamps", htype="text")
    ds.create_tensor("data", htype="generic")

# Create timeline of commits
timeline = []

# Commit 1: Initial state
with ds:
    ds.timestamps.append("2024-01-01")
    ds.data.append(100)
commit_1 = ds.commit("Initial: 1 sample")
timeline.append(("2024-01-01", commit_1, len(ds)))

# Commit 2: Add more
with ds:
    ds.timestamps.append("2024-01-02")
    ds.data.append(200)
commit_2 = ds.commit("Added: 2 samples")
timeline.append(("2024-01-02", commit_2, len(ds)))

# Commit 3: More additions
with ds:
    ds.timestamps.append("2024-01-03")
    ds.data.append(300)
commit_3 = ds.commit("Added: 3 samples")
timeline.append(("2024-01-03", commit_3, len(ds)))

# Time travel through history
print("=== Time Travel Demo ===\n")
for date, commit_id, sample_count in timeline:
    ds.checkout(commit_id)
    print(f"Date: {date}")
    print(f"Commit: {commit_id[:8]}...")
    print(f"Samples: {sample_count}")
    print(f"Data: {ds.data.numpy()}")
    print(f"Timestamps: {[ts.data()['text'] for ts in ds.timestamps]}")
    print()

# Return to present
ds.checkout("HEAD")
print(f"Back to present: {len(ds)} samples")
```

---

## Key Takeaways

### For Weekly Backups:
- ✅ Use `deeplake.copy()` to preserve all versions
- ✅ Automate with cron/task scheduler
- ✅ Implement retention policy (keep last 4-8 backups)
- ✅ Store backups on separate storage

### For Cloud Sync:
- ✅ Use `ds.push()` to sync local → cloud
- ✅ Use `deeplake.pull()` to sync cloud → local
- ✅ Always commit changes before pushing
- ✅ Use bidirectional sync to prevent conflicts

### For Migration:
- ✅ Use `deeplake.copy()` for one-time migration
- ✅ **This preserves ALL version history**
- ✅ Verify migration by comparing commit counts
- ✅ After migration, use push/pull for ongoing sync

---

## Important Notes

1. **Version 4+ Only**: All methods shown are for DeepLake 4.x+
2. **Authentication**: Ensure cloud credentials are configured before syncing
3. **Network**: Large datasets require stable, high-bandwidth connections
4. **Storage Costs**: Cloud storage costs scale with dataset size
5. **Testing**: Always test backup/restore procedures before relying on them

---

## Troubleshooting

### Issue: "Dataset already exists" error
**Solution**: Use `overwrite=True` in `deeplake.copy()` or `deeplake.pull()`

### Issue: Slow sync/backup
**Solution**: Increase `num_workers` parameter or use `multiprocessing=True`

### Issue: Version history not preserved
**Solution**: Always use `deeplake.copy()` for migration, not manual file copying

### Issue: Authentication errors
**Solution**: Verify cloud credentials are properly set in environment variables

---

**Last Updated**: Based on DeepLake 4+ API (2024-2025)

