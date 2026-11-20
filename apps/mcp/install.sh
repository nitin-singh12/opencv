#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# See the LICENSE file distributed with this application for license terms.

# Install/check all nine tools and generate settings for local stdio MCP clients.
set -euo pipefail
if [ -n "${PYTHON:-}" ]; then
  INSTALL_PYTHON="$PYTHON"
elif command -v python3.12 >/dev/null 2>&1; then
  INSTALL_PYTHON=python3.12
else
  INSTALL_PYTHON=python3
fi
exec "$INSTALL_PYTHON" "$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)/scripts/install_mcp.py" "$@"
