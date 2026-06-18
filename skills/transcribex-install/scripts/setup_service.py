#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def main() -> int:
    args = parse_args()
    try:
        status = request_json(args.base_url, "/v1/setup/status")
        if args.show:
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0

        profile_id = args.profile
        if args.auto and not profile_id:
            profile_id = recommended_profile_id(status)
        if not profile_id and not args.asr_model:
            print("Choose --auto, --profile, or --asr-model.", file=sys.stderr)
            return 2

        payload: dict[str, Any] = {
            "setup_complete": True,
        }
        if profile_id:
            payload["profile_id"] = profile_id
        optional = {
            "asr_model": args.asr_model,
            "device": args.device,
            "vad_model": args.vad_model,
            "punc_model": args.punc_model,
            "spk_model": args.spk_model,
            "hub": args.hub,
            "batch_size_s": args.batch_size_s,
            "api_key": args.set_api_key,
            "max_upload_mb": args.max_upload_mb,
            "preload_model": args.preload_model,
            "keep_uploads": args.keep_uploads,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        result = request_json(args.base_url, "/v1/setup/apply", method="POST", payload=payload, api_key=args.api_key)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"TranscribeX setup API error {exc.code}: {detail}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Failed to connect to TranscribeX at {args.base_url}: {exc.reason}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure a running TranscribeX v2 service through its setup API.")
    parser.add_argument("--base-url", default=os.environ.get("TRANSCRIBEX_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.environ.get("TRANSCRIBEX_API_KEY"))
    parser.add_argument("--show", action="store_true", help="Only print setup status and recommendations.")
    parser.add_argument("--auto", action="store_true", help="Apply the first server-recommended profile.")
    parser.add_argument("--profile", help="Profile ID to apply, such as cpu-balanced or gpu-balanced.")
    parser.add_argument("--asr-model", help="Override ASR model.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Override runtime device.")
    parser.add_argument("--vad-model", help="Set VAD model. Use an empty string to disable.")
    parser.add_argument("--punc-model", help="Set punctuation model. Use an empty string to disable.")
    parser.add_argument("--spk-model", help="Set speaker model. Use an empty string to disable.")
    parser.add_argument("--hub", help="Optional model hub name.")
    parser.add_argument("--batch-size-s", type=int, help="FunASR batch_size_s.")
    parser.add_argument("--set-api-key", help="Persist a new TranscribeX bearer token.")
    parser.add_argument("--max-upload-mb", type=int, help="Maximum uploaded file size in MB.")
    parser.add_argument("--preload-model", dest="preload_model", action="store_true", default=None)
    parser.add_argument("--no-preload-model", dest="preload_model", action="store_false")
    parser.add_argument("--keep-uploads", dest="keep_uploads", action="store_true", default=None)
    parser.add_argument("--discard-uploads", dest="keep_uploads", action="store_false")
    return parser.parse_args()


def request_json(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}{path}",
        method=method,
        data=data,
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset))


def recommended_profile_id(status: dict[str, Any]) -> str:
    profiles = status.get("recommended_profiles") or []
    if not profiles:
        raise urllib.error.URLError("setup API returned no recommended profiles")
    return profiles[0]["id"]


if __name__ == "__main__":
    raise SystemExit(main())
