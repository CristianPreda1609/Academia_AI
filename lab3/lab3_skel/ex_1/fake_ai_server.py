from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import json
import time
import random

class FakeAIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        time.sleep(2)
        ln = random.randint(1, 100)

        response = {
            "response": f"Hello from Fake AI. Your lucky number is {ln}"
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        self.wfile.write(
            json.dumps(response).encode("utf-8")
        )


server = ThreadingHTTPServer(
    ("localhost", 8000),
    FakeAIHandler
)
server.daemon_threads = True

print("Server started at http://localhost:8000")

server.serve_forever()