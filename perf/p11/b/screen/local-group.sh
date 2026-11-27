#!/usr/bin/env bash
# Reserve the Stream B queue for one short operator sweep. Each measurement still obtains the shared GPU lock.
set -euo pipefail
exec {group_fd}>build/p11/stream-b.lock
flock "$group_fd"
export B_LOCAL_GROUP_FD=$group_fd
exec "$@"
