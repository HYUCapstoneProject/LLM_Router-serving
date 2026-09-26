"""Read-only HTTP health/model checks. Does not submit training or switch models."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-model-version", required=True)
    parser.add_argument("--batch-id", help="Optional existing batch; never creates a batch")
    args = parser.parse_args()
    token = os.environ.get("LLM_ROUTER_INTERNAL_API_TOKEN", "")
    if not token:
        parser.error("LLM_ROUTER_INTERNAL_API_TOKEN is required in the process environment")
    endpoints = ["/health", "/router/model/current"]
    if args.batch_id:
        endpoints.append("/retraining-runs/" + urllib.parse.quote(args.batch_id, safe=""))
    outputs = {}
    try:
        for endpoint in endpoints:
            request = urllib.request.Request(
                args.base_url.rstrip("/") + endpoint,
                headers={"Authorization": "Bearer " + token},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                outputs[endpoint] = json.load(response)
    except (urllib.error.URLError, ValueError):
        # Never echo request headers, credentials or server-provided error details.
        print(json.dumps({"http_ready": False, "error": "endpoint unavailable or invalid JSON"}))
        return 1
    current = outputs["/router/model/current"]
    ready = (
        outputs["/health"].get("status") == "ok"
        and current.get("status") == "inSync"
        and (current.get("serving") or {}).get("modelVersion") == args.expected_model_version
        and (current.get("desired") or {}).get("modelVersion") == args.expected_model_version
        and current.get("lastLoadError") is None
    )
    print(json.dumps({"http_ready": ready, "pipeline_verified": False, "responses": outputs}))
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
