#!/usr/bin/env python3
"""
Simple Baseline Cleanup - Remove duplicate baseline files
"""

import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def cleanup_duplicate_baselines():
    """Clean up duplicate baseline files, keeping only the latest"""
    
    print("🧹 VariBAD Baseline Cleanup")
    print("============================")
    print()
    
    baselines_dir = Path("tests/baselines")
    
    if not baselines_dir.exists():
        print("❌ No baselines directory found")
        return False
    
    # Find all baseline files
    baseline_files = list(baselines_dir.glob("baseline_*_dataset.json"))
    
    if len(baseline_files) <= 1:
        print("✅ Only one or no baseline files found - no cleanup needed")
        return True
    
    print(f"📄 Found {len(baseline_files)} baseline files:")
    
    # Sort by modification time (newest first)
    baseline_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Show all files with details
    for i, bf in enumerate(baseline_files):
        stat = bf.stat()
        modified = datetime.fromtimestamp(stat.st_mtime)
        status = "KEEP (latest)" if i == 0 else "REMOVE (duplicate)"
        print(f"   {bf.name} - {stat.st_size:,} bytes - {modified.strftime('%Y-%m-%d %H:%M')} - {status}")
    
    latest_file = baseline_files[0]
    old_files = baseline_files[1:]
    
    print()
    print(f"📋 PLAN:")
    print(f"   Keep: {latest_file.name}")
    print(f"   Remove: {len(old_files)} duplicate files")
    print(f"   Space saved: {sum(f.stat().st_size for f in old_files):,} bytes")
    
    # Verify files are actually duplicates by checking checksums
    print()
    print("🔍 Verifying files are identical...")
    
    checksums = {}
    for bf in baseline_files:
        try:
            with open(bf, 'r') as f:
                content = f.read()
                # Extract checksum from JSON
                import re
                match = re.search(r'"checksum":\s*"([^"]+)"', content)
                if match:
                    checksums[bf] = match.group(1)
                    print(f"   {bf.name}: {match.group(1)[:16]}...")
        except Exception as e:
            print(f"   Error reading {bf.name}: {e}")
            return False
    
    # Check if all checksums are identical
    unique_checksums = set(checksums.values())
    if len(unique_checksums) != 1:
        print("⚠️  WARNING: Files have different checksums!")
        print("   Manual review required - not proceeding with cleanup")
        return False
    
    print("✅ All files have identical checksums - safe to remove duplicates")
    
    # Ask for confirmation
    print()
    response = input("Proceed with cleanup? (y/N): ")
    if response.lower() != 'y':
        print("Cleanup cancelled")
        return False
    
    # Create backup directory
    backup_dir = Path(".cleanup_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Create git backup tag
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        subprocess.run(["git", "tag", f"before-baseline-cleanup-{timestamp}"], 
                     check=True, capture_output=True)
        print(f"✅ Created git backup tag: before-baseline-cleanup-{timestamp}")
    except:
        print("⚠️  Could not create git tag (may not be in git repo)")
    
    # Remove old files
    print()
    print("🗑️  Removing duplicate files...")
    
    removed_count = 0
    for old_file in old_files:
        try:
            # Create backup first
            backup_name = f"{old_file.name}.{timestamp}"
            shutil.copy2(old_file, backup_dir / backup_name)
            
            # Remove original
            old_file.unlink()
            print(f"   ✅ Removed {old_file.name}")
            removed_count += 1
        except Exception as e:
            print(f"   ❌ Failed to remove {old_file.name}: {e}")
    
    # Commit to git
    try:
        subprocess.run(["git", "add", "-A"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Clean up duplicate baseline files"], 
                     check=True, capture_output=True)
        print("✅ Changes committed to git")
    except:
        print("⚠️  Git commit failed (may not be in git repo)")
    
    print()
    print("🎉 CLEANUP COMPLETE!")
    print("====================")
    print(f"✅ Removed {removed_count} duplicate baseline files")
    print(f"✅ Kept latest: {latest_file.name}")
    print(f"✅ Backups saved in: {backup_dir}")
    
    if removed_count > 0:
        space_saved = sum(f.stat().st_size for f in old_files if not f.exists())
        print(f"✅ Space saved: {space_saved:,} bytes")
    
    print()
    print("📋 Recovery (if needed):")
    print(f"   git reset --hard before-baseline-cleanup-{timestamp}")
    
    return True

if __name__ == "__main__":
    cleanup_duplicate_baselines()