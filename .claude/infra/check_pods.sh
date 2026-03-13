#!/bin/bash
# Check all running RunPod pods and report status.
# Flags idle pods (0% GPU) that should be terminated.

RUNPOD_API_KEY="$(grep apikey ~/.runpod/config.toml 2>/dev/null | cut -d'"' -f2)"
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: No RunPod API key found in ~/.runpod/config.toml"
    exit 1
fi

RESULT=$(curl -s -H "Content-Type: application/json" \
  -d '{"query":"query { myself { pods { id name desiredStatus machine { gpuDisplayName costPerHr } runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } } } } }"}' \
  "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY")

echo "$RESULT" | python3 -c "
import json, sys
data = json.load(sys.stdin)
pods = data.get('data', {}).get('myself', {}).get('pods', [])
if not pods:
    print('No running pods.')
    sys.exit(0)
for p in pods:
    rt = p.get('runtime') or {}
    gpus = rt.get('gpus', [{}])
    gpu_util = gpus[0].get('gpuUtilPercent', -1) if gpus else -1
    mem_util = gpus[0].get('memoryUtilPercent', -1) if gpus else -1
    uptime_h = rt.get('uptimeInSeconds', 0) / 3600
    cost_so_far = uptime_h * (p.get('machine', {}).get('costPerHr', 0))
    status = 'IDLE' if gpu_util == 0 else 'ACTIVE'
    print(f\"{p['id']}  {p['name']:25s}  {p['machine']['gpuDisplayName']:15s}  GPU:{gpu_util:3d}%  MEM:{mem_util:3d}%  {uptime_h:.1f}h  \${cost_so_far:.2f}  [{status}]\")
    if gpu_util == 0:
        print(f'  ^^^ WARNING: Pod {p[\"id\"]} is IDLE. Consider terminating.')
"
