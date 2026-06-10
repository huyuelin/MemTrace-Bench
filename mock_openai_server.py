#!/usr/bin/env python3
"""Mock OpenAI-compatible server for testing experiment pipeline.

Run with: python mock_openai_server.py
Then set OPENAI_API_KEY=test and OPENAI_BASE_URL=http://127.0.0.1:8888/v1
"""
import json
import http.server
import socketserver

class MockHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.endswith('/chat/completions'):
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            messages = data.get('messages', [])
            # Generate a deterministic patch response
            patch = """--- a/README.md
+++ b/README.md
@@ -1,4 +1,4 @@
-# gitignore
+# .gitignore
 
 [![Build Status](https://github.com/github/gitignore/workflows/Tests/badge.svg)](https://github.com/github/gitignore/actions?query=workflow%3ATests)
"""
            response = {
                "choices": [{
                    "message": {
                        "content": patch,
                        "role": "assistant"
                    }
                }]
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logs

if __name__ == '__main__':
    with socketserver.TCPServer(("127.0.0.1", 9999), MockHandler) as httpd:
        print("Mock OpenAI server running on http://127.0.0.1:9999")
        httpd.serve_forever()
