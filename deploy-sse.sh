#!/bin/bash

# Deploy SSE streaming responses endpoint for TID

# Run from your server after downloading responses_endpoint.py

set -e

TID_DIR=”/home/josh/The-Inference-Difference”
DEST=”$TID_DIR/inference_difference/responses_endpoint.py”

# 1. Copy the file

echo “Copying responses_endpoint.py to $DEST…”
cp responses_endpoint.py “$DEST”

# 2. Patch app.py if not already patched

if ! grep -q “responses_endpoint” “$TID_DIR/inference_difference/app.py”; then
echo “Patching app.py…”

```
# Add import
sed -i '/from inference_difference.translation_shim import translate_request/a from inference_difference.responses_endpoint import register_responses_endpoint' "$TID_DIR/inference_difference/app.py"

# Add registration at end
cat >> "$TID_DIR/inference_difference/app.py" << 'APPEOF'
```

# OpenAI Responses API (SSE streaming for OpenClaw)

register_responses_endpoint(app, chat_completions)
APPEOF
echo “app.py patched.”
else
echo “app.py already has responses_endpoint import.”
fi

# 3. Restart TID

echo “Restarting TID…”
sudo systemctl restart inference-difference
sleep 3

# 4. Test streaming

echo “Testing SSE streaming…”
curl -s -N -X POST http://localhost:7437/v1/responses   
-H “Content-Type: application/json”   
-d ‘{“model”:“auto”,“input”:“Say hello briefly”,“stream”:true}’ | head -20

echo “”
echo “If you see ‘event: response.created’ and ‘event: response.output_text.delta’ above, SSE is working!”
echo “Now restart OpenClaw: systemctl –user restart openclaw-gateway”
