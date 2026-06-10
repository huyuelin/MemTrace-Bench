#!/usr/bin/env python3
"""
Update sample_sequences.json to add 'repo' field to each sequence.
The 'repo' field is extracted from 'repo_url' in format 'owner/repo_name'.
"""
import json
import os

# Read the JSON file
json_path = os.path.join(os.path.dirname(__file__), "../data/sample_sequences.json")
with open(json_path) as f:
    sequences = json.load(f)

# Add 'repo' field to each sequence
for seq in sequences:
    repo_url = seq.get("repo_url", "")
    # Extract owner/repo from URL like "https://github.com/test-org/repo-001"
    if repo_url:
        # Remove protocol and domain
        path = repo_url.split("github.com/")[-1] if "github.com/" in repo_url else ""
        seq["repo"] = path
    else:
        seq["repo"] = "unknown/unknown"

# Write updated JSON back
with open(json_path, "w") as f:
    json.dump(sequences, f, indent=2)

print(f"Updated {len(sequences)} sequences with 'repo' field")
print("Sample repo values:")
for seq in sequences[:3]:
    print(f"  {seq['sequence_id']}: repo={seq.get('repo')}")
