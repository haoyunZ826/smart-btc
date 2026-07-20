#!/bin/bash
# Preview the dashboard locally at http://localhost:8790
cd "$(dirname "$0")/../site" || exit 1
echo "serving site/ at http://localhost:8790"
python3 -m http.server 8790
