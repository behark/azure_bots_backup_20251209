#!/usr/bin/env bash
# Backup all state files before major changes

set -euo pipefail

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Backing up state files..."

# Counter for backed up files
backup_count=0

# Find and backup all state files
while IFS= read -r file; do
    if [ -f "$file" ]; then
        rel_path="${file#./}"
        dir_path="$(dirname "$rel_path")"
        mkdir -p "$BACKUP_DIR/$dir_path"
        cp "$file" "$BACKUP_DIR/$rel_path"
        ((backup_count++))
    fi
done < <(find . -name "*_state.json" -type f 2>/dev/null)

# Backup configs
if [ -f "global_config.json" ]; then
    cp global_config.json "$BACKUP_DIR/" 2>/dev/null || true
    ((backup_count++))
fi

if [ -f ".env" ]; then
    cp .env "$BACKUP_DIR/.env.backup" 2>/dev/null || true
    ((backup_count++))
fi

# Backup watchlists
find . -name "*watchlist.json" -type f | while read -r file; do
    if [ -f "$file" ]; then
        rel_path="${file#./}"
        dir_path="$(dirname "$rel_path")"
        mkdir -p "$BACKUP_DIR/$dir_path"
        cp "$file" "$BACKUP_DIR/$rel_path"
        ((backup_count++))
    fi
done

if [ $backup_count -eq 0 ]; then
    echo "⚠️  No files found to backup"
    rmdir "$BACKUP_DIR"
    exit 0
fi

echo "✅ Backup complete: $BACKUP_DIR"
echo "💾 Files backed up: $backup_count"
echo "📊 Size: $(du -sh "$BACKUP_DIR" 2>/dev/null | cut -f1 || echo 'unknown')"
echo ""
echo "To restore from backup:"
echo "  cp -r $BACKUP_DIR/* ."
