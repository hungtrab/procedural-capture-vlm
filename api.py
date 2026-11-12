import asyncio
import json
import os
import random
import re
import subprocess
import time

# AI Gateway, reached via the x2pagentd forward proxy that runs on GenAI cluster
# nodes. No API key needed: identity travels with the request (x2pagentd vouches
# for the caller, exactly as the AI Gateway quickstart describes for corp hosts).
# The Python OpenAI/httpx SDK gets connection-reset by x2pagentd (peer-process
# attestation only allowlists curl), so we shell out to curl.
#
# Every gateway is <name>.ai-gateway.x2p.facebook.net, where the FIRST LABEL is
# the registered upstream's name and the path is that provider's own native API,
# forwarded untouched. A name with no upstream answers 421 Misdirected Request,
# which makes the live set discoverable by probing.
#
# The Plugboard passthrough pipelines we used before (plugboard.ai-gateway...,
# /codex/passthrough + /claude_code/passthrough) now answer 403 "service ACL
# blocklisted the request" for EVERY model on both routes -- verified 2026-08-07
# across gpt-5/5.4/5.5/5.3-codex/grok-4 and claude-haiku-4.5. They are dead ends;
# the per-upstream gateways below are the sanctioned replacement.
_AI_GATEWAY_DOMAIN = "ai-gateway.x2p.facebook.net"

# gpt-5.5, served by the azure_openai upstream registered as `azure-codex-gpt-5-5`
# (see its cinc config: provider=azure_openai on dev-infra-ncusn2). The path is
# Azure's own /openai/v1/responses, i.e. the Responses API -- the same wire shape
# the old route spoke, so _to_responses_payload/_responses_text are unchanged.
# This upstream also exposes /openai/v1/chat/completions, but we stay on Responses
# because reasoning.effort and the starvation recovery below depend on it.
RESPONSES_URL = f"http://azure-codex-gpt-5-5.{_AI_GATEWAY_DOMAIN}/openai/v1/responses"

# Models served by a DIFFERENT azure upstream than the default above. Both
# deployments currently answer on either host -- gpt-5.6-sol resolves through
# azure-codex-gpt-5-5 and vice versa, verified 2026-08-13 -- but rate limits are
# configured per upstream registration, so a model must go to its own gateway or
# its traffic is metered against the wrong budget. gpt-5.6-sol's registration
# (azure-codex-gpt-5-6-sol) sets max_rpm=125 / max_rp10m=600, i.e. 1 req/s
# sustained: roughly 4x tighter than the anthropic limit that cost 34% of calls
# in job 361470. Pace it (openai_shim.py's SHIM_RATE_LIMIT) before using it at
# any volume.
_RESPONSES_HOSTS = {
    "gpt-5.6-sol": "azure-codex-gpt-5-6-sol",
    # A SECOND deployment of the same frontier Codex route, registered separately.
    # Its upstream declares the identical max_rpm=125 / max_rp10m=600 but with
    # `enforced=False`, and that difference is the whole point: measured
    # 2026-09-24 from a cpu node, 96 concurrent requests returned 1,046/min with
    # zero rejections, p50 1.1s / p95 1.7s -- eighteen times what the enforced
    # -sol route sustains.
    #
    # It is NOT an alias for -sol. Both the model name and the host must be the
    # luna pair: `gpt-5.6` or `gpt-5.6-sol` against this host return
    # DeploymentNotFound, as does `gpt-5.6-luna` against the default host. Keep
    # the distinct name so a results column that used this route says so -- the
    # two are different deployments and may not be the same weights.
    "gpt-5.6-luna": "azure-codex-gpt-5-6-luna",
}


def responses_url(model=None):
    """Responses endpoint for `model`, defaulting to the gpt-5.5 upstream.

    Note the host label is the UPSTREAM's registered name, not the model's:
    `azure-codex-gpt-5-6` is unregistered and answers 421, while
    `azure-codex-gpt-5-6-sol` is the live one.
    """
    host = _RESPONSES_HOSTS.get(model or "")
    return (f"http://{host}.{_AI_GATEWAY_DOMAIN}/openai/v1/responses"
            if host else RESPONSES_URL)

# x2pagentd's address. Pods and sandboxes run the agent on the host NAT address
# rather than loopback and publish the full URL in X2P_PROXY_URL; the hardcoded
# localhost this file used before simply got "connection refused" there. Prefer the
# published URL, then a per-job X2P_PORT (compute nodes spawn their own agent on a
# kernel-picked port when the job opts in with `--export=ALL,WITH_X2P=1`), then the
# shared login-node default.
PROXY = os.environ.get("X2P_PROXY_URL") or f"http://localhost:{os.environ.get('X2P_PORT', '10054')}"

# Attribution for the gateway's logs; both x-meta-ai-gateway-* headers are optional
# and stripped before the request leaves Meta.
CALLING_PRODUCT = os.environ.get("AI_GATEWAY_CALLING_PRODUCT", "userrl")

# Default model. Names on the AI Gateway registry use dots, not dashes
# (gpt-5.5, not the old gpt-5-5) -- normalize_model() rewrites the old spellings
# so stale env vars and call sites keep working.
DEFAULT_MODEL = "gpt-5.5"

# The routable OpenAI-family models. Every other name (gpt-5, gpt-5.4, grok-4,
# the codex variants) answers 404 DeploymentNotFound, and most sibling gateways
# have no live deployment behind them -- `azure-codex-gpt-5-6` in particular is
# unregistered and answers 421, which is why the entry below carries the `-sol`
# suffix on both the model and its upstream. Unroutable names are deliberately
# NOT aliased to a live model: silently swapping model identity would corrupt
# eval results, so they fail loudly instead.
#
# gpt-5.6-sol is heavily rate limited (see _RESPONSES_HOSTS). Its MetaGen
# entitlement name falls out of the existing rule unchanged:
# _metagen_model("gpt-5.6-sol") -> "gpt-5-6-sol-genai-responses".
AVAILABLE_MODELS = [
    "gpt-5.5",
    "gpt-5.6-sol",
]

# Claude lives on the Anthropic-native /v1/messages route of the `anthropic`
# gateway, not the Responses route above. Reach it via chat_messages()/
# chat_messages_async(); is_messages_model() routes automatically. Unlike the dead
# Plugboard pipeline, this host REQUIRES an anthropic-version header (it answers
# 400 "anthropic-version: header is required" without one).
MESSAGES_URL = f"http://anthropic.{_AI_GATEWAY_DOMAIN}/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# This upstream spells versions with a DASH: claude-opus-4-7 resolves, while the
# dotted claude-opus-4.7 this file used to send 404s with "Did you mean
# claude-opus-4-7?". That spelling -- not availability -- is the whole reason the
# old default looked dead, so normalize_model() rewrites the dots and every model
# keeps its exact identity. Only claude-opus-4.1 is genuinely gone (404).
DEFAULT_MESSAGES_MODEL = "claude-opus-5"
ANTHROPIC_MESSAGES_MODELS = [
    "claude-opus-5", "claude-opus-4.8", "claude-opus-4.7", "claude-opus-4.6",
    "claude-opus-4.5",
    "claude-sonnet-5", "claude-sonnet-4.6", "claude-sonnet-4.5",
    "claude-haiku-4.5",
]

# Gemini lives on the `confucius-vertex-gemini` gateway, a vertex-provider upstream
# (cinc config: provider=vertex(account="confucius-gcp-dev-xgmwf")). It speaks
# Google's own Vertex API -- contents/parts instead of messages, systemInstruction
# instead of a system turn, generationConfig instead of top-level knobs -- so it
# gets its own payload builder and extractor below. is_gemini_model() routes
# chat()/chat_async()/chat_parse() automatically.
#
# The `account` from the cinc config doubles as the GCP project in the URL path.
# It is not interchangeable: the other Vertex gateways' project (devai-mea-egeit)
# answers 403 "Permission 'aiplatform.endpoints.predict' denied" on this host.
GEMINI_PROJECT = os.environ.get("GEMINI_PROJECT", "confucius-gcp-dev-xgmwf")
GEMINI_LOCATION = os.environ.get("GEMINI_LOCATION", "global")
GEMINI_HOST = f"http://confucius-vertex-gemini.{_AI_GATEWAY_DOMAIN}"
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

# Verified callable 2026-08-07 by actually generating with each. Note the version
# separator CHANGES between generations: the 3.0 generation uses a dash
# (gemini-3-flash-preview) while 3.1+ use dots (gemini-3.5-flash) -- so no
# normalization is applied to gemini names, both spellings are real.
#
# This gateway carries strictly more than the `vertex`/`playground` ones, which
# serve only the 2.5 family. Two discovery methods were needed and neither is
# complete alone: /v1beta1/publishers/google/models lists the catalog (but omits
# the aliases and the image/computer-use deployments, and over-reports 7 entries
# this project cannot call), while probing finds those but misses every dotted 3.x
# name. Re-check with both.
GEMINI_MODELS = [
    "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview", "gemini-3.1-flash-lite",
    "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image",
    "gemini-3-flash-preview", "gemini-3-pro-image",
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
    "gemini-2.5-flash-image", "gemini-2.5-computer-use-preview-10-2025",
    "gemini-flash-latest", "gemini-flash-lite-latest",
]

# Not reachable from this module: glm-5.2, claude-opus-4.1, and every OpenAI name
# but gpt-5.5. Calls naming them fail loudly at the server -- deliberately not
# aliased to a survivor, since silently swapping model identity would corrupt eval
# results. Also present on the Gemini gateway but needing a different protocol:
# gemini-embedding-2 (:embedContent, 3072 dims), the *-tts models (audio response
# config) and gemini-live-* (bidi WebSocket).

# Plugboard/passthrough reasoning models accept effort in
# {none, low, medium, high, xhigh} but reject 'minimal' (an OpenAI-only tier)
# with HTTP 400. Normalize once here so call sites can keep passing 'minimal'.
_REASONING_EFFORT_ALIASES = {"minimal": "low"}

# Pre-migration spellings -> AI Gateway registry names. The old REST proxy used
# dashes for the minor version; the gateway uses dots.
_RENAMED_MODELS = {
    "gpt-5-5": "gpt-5.5",
    "gpt-5-4": "gpt-5.4",
    "gpt-5-4-pro": "gpt-5.4-pro",
    "gpt-5-2": "gpt-5.2",
    "gpt-5-1-codex": "gpt-5.1-codex",
    "gpt-5-2-codex": "gpt-5.2-codex",
    "gpt-5-3-codex": "gpt-5.3-codex",
}

# Reasoning tokens are billed against max_output_tokens on the Responses API, so a
# caller's max_tokens (which historically meant "visible text budget") can be fully
# consumed by thinking, returning status=incomplete with an empty message. Add
# headroom up front, and escalate on a starved response.
_REASONING_HEADROOM = 8192
_MAX_OUTPUT_TOKENS_CAP = 65536


def normalize_model(model):
    """Map stale model spellings to their AI Gateway registry names.

    e.g. gpt-5-5-genai-responses -> gpt-5.5, claude-opus-4.7 -> claude-opus-4-7.
    Empty/None -> DEFAULT_MODEL.
    """
    if not model:
        return DEFAULT_MODEL
    for suffix in ("-genai-responses", "-genai"):
        if model.endswith(suffix):
            model = model[: -len(suffix)]
            break
    model = _RENAMED_MODELS.get(model, model)
    # The two upstreams disagree on how to punctuate a version: Azure's deployment
    # is `gpt-5.5` with a dot, Anthropic's is `claude-opus-4-7` with a dash, and
    # each 404s the other's spelling. Fix it here so call sites, env vars and
    # eval configs can keep using one convention.
    if model.startswith("claude-"):
        model = model.replace(".", "-")
    return model


def normalize_reasoning_effort(effort):
    if effort is None:
        return None
    return _REASONING_EFFORT_ALIASES.get(effort, effort)


def is_messages_model(model):
    """True iff ``model`` belongs on the Anthropic-native /v1/messages route
    rather than the Responses route -- i.e. every Claude model.

    Matched on the family prefix rather than membership of
    ANTHROPIC_MESSAGES_MODELS, so a retired name like claude-opus-4.7 still routes
    to Anthropic and fails with its own "Did you mean claude-opus-4-7?" instead of
    being sent to Azure and coming back as an unrelated DeploymentNotFound.
    """
    return normalize_model(model).startswith("claude-")


def is_gemini_model(model):
    """True iff ``model`` belongs on the Vertex :generateContent route.

    Prefix-matched for the same reason as is_messages_model: an unknown gemini-*
    name should fail with Vertex's own "Publisher model ... not found" rather than
    Azure's unrelated DeploymentNotFound.
    """
    return normalize_model(model).startswith("gemini")


def merge_alternating(messages):
    """Coerce a messages list into strict user/assistant alternation for the
    /v1/messages endpoint, which rejects leading-assistant or consecutive
    same-role turns. Drops leading assistant turns and joins consecutive
    same-role contents with a newline. (System messages should be removed by the
    caller and passed via the top-level ``system`` field.)"""
    out = []
    for m in messages:
        if not out and m["role"] == "assistant":
            continue
        content = m["content"] or ""
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {**out[-1], "content": out[-1]["content"] + "\n" + content}
        else:
            out.append({"role": m["role"], "content": content})
    return out


def _curl_base(url, timeout, extra_headers=()):
    """Curl argv for a POST to AI Gateway through the x2pagentd forward proxy.

    Body is read from stdin (--data-binary @-). Factored out so every transport
    path shares one definition. The old x-x2pagentd-inject-cat header is gone with
    the Plugboard pipelines it authenticated -- the per-upstream gateways authorize
    the caller's own identity, so no token is injected.
    """
    argv = [
        "curl", "-sS", "--max-time", str(timeout),
        "-X", "POST", url,
        "--proxy", PROXY,
        "-H", "Content-Type: application/json",
        "-H", f"x-meta-ai-gateway-calling-product: {CALLING_PRODUCT}",
    ]
    for header in extra_headers:
        argv += ["-H", header]
    return argv + ["--data-binary", "@-"]


# The Anthropic upstream rejects a request that omits this; the Azure one has no
# equivalent requirement, so it stays off the Responses path.
_MESSAGES_HEADERS = (f"anthropic-version: {ANTHROPIC_VERSION}",)


def _to_responses_payload(
    messages, model, max_tokens, response_format=None, reasoning_effort=None,
    tools=None, tool_choice=None
):
    """Build a /v1/responses body from chat-completions-style arguments.

    Field mapping: messages -> input (the 'system' role is accepted inside input,
    so no split into `instructions` is needed), max_tokens -> max_output_tokens,
    reasoning_effort -> reasoning.effort, response_format -> text.format.
    ``seed`` has no equivalent and is dropped by callers; sending either it or a
    literal max_tokens is a hard 400 ("Unknown parameter").
    """
    effort = normalize_reasoning_effort(reasoning_effort)
    # max_output_tokens covers reasoning AND visible text, so reasoning calls need headroom
    # or they come back starved. With effort="none" nothing is spent thinking, and the
    # headroom silently becomes 5x the OUTPUT budget the caller asked for.
    #
    # That is not hypothetical: mirrorbench:616 asks for 2048 and was being given 10,240, so
    # the mirror-assistant wrote 19k-char turns. mirrorbench appends each reply to
    # simulation_turns and resends the lot, so the payload compounded to 49KB and stopped
    # fitting in the gateway's 60s ingress cap -- job 337693 died at step 2 that way, and the
    # dump shows 10,177 -> 13,300 -> 19,226 char replies in one episode. Run 323488, which
    # completed 28h, tops out at 7,265 chars: a plain 2048-token ceiling.
    #
    # Only "none" is treated as non-reasoning. effort=None means the field is omitted and a
    # reasoning model may still think by default, so that case keeps the headroom. If "none"
    # is ever ignored upstream, the _starved recovery below still doubles the budget.
    budget = max_tokens if effort == "none" else max_tokens + _REASONING_HEADROOM
    payload = {
        "model": normalize_model(model),
        "input": messages,
        "max_output_tokens": min(budget, _MAX_OUTPUT_TOKENS_CAP),
    }
    # Tool calling, for using a hosted model as the AGENT rather than the
    # simulator. eval.py drives the agent with tools=[schema] and
    # tool_choice="required" and discards any reply without a structured call,
    # so this is the difference between a usable agent and a 100% failure rate.
    #
    # The Responses API flattens the tool spec: chat-completions nests it as
    # {"type":"function","function":{name,description,parameters}} while this
    # route wants those keys at the top level. Sending the nested form is a 400.
    if tools:
        payload["tools"] = [
            ({"type": "function", **t["function"]}
             if isinstance(t, dict) and "function" in t else t)
            for t in tools
        ]
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    if effort is not None:
        payload["reasoning"] = {"effort": effort}
    if response_format is not None:
        payload["text"] = {"format": response_format}
    return payload


def _responses_text(data):
    """Extract the assistant text from a /v1/responses body.

    ``output`` is a list of items; reasoning items carry no visible text, so we
    take the text of every output_text block on the message items.
    """
    parts = [
        c.get("text", "")
        for item in data.get("output", [])
        if item.get("type") == "message"
        for c in item.get("content", [])
        if c.get("type") == "output_text"
    ]
    return "".join(parts)


def _responses_tool_calls(data):
    """Pull function_call items out of a /v1/responses body, in OpenAI
    chat-completions shape, so a caller can hand them straight to a client that
    expects `message.tool_calls`."""
    out = []
    for item in data.get("output", []):
        if item.get("type") != "function_call":
            continue
        out.append({
            "id": item.get("call_id") or item.get("id") or f"call_{len(out)}",
            "type": "function",
            "function": {"name": item.get("name"),
                         "arguments": item.get("arguments") or "{}"},
        })
    return out


def _starved(data):
    """True if the model burned its whole budget on reasoning and emitted no text."""
    return (
        data.get("status") == "incomplete"
        and (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens"
    )


async def _post_async(url, body, timeout, extra_headers=()):
    """Async POST to AI Gateway. Returns (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *_curl_base(url, timeout, extra_headers),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(body.encode())
    return proc.returncode, out.decode(), err.decode()


def _mentions_reasoning(data) -> bool:
    return "reasoning" in json.dumps(data).lower()


# ── MetaGen fallback ──────────────────────────────────────────────────────────────────
# The AI Gateway goes through multi-hour degraded windows where it holds requests open and
# cuts them at its 60s ingress cap. Job 336216: 408 failed attempts, 96% at exactly 60.2s,
# payloads from 752B to 37KB -- the outage is indiscriminate, so retrying the same host is
# futile. MetaGen is an unrelated OpenAI-compatible endpoint that survives those windows.
#
# Past _FALLBACK_AFTER failed attempts every further retry tries MetaGen FIRST and then
# still falls through to the gateway, so whichever recovers first serves the call. Gateway
# logging is deliberately untouched: the per-attempt lines keep printing even while MetaGen
# is covering, so a degraded window stays visible instead of being silently papered over.
#
# Needs LLAMA_API_KEY in the environment. Export it before `ray start` in run_rl.sh so the
# srun'd worker nodes inherit it via --export=ALL, same as OPENAI_MODEL_NAME.
METAGEN_BASE_URL = "https://api.llama.com/compat/v1/"
_FALLBACK_AFTER = int(os.getenv("API_FALLBACK_AFTER", "3"))
# MetaGen's compat endpoint takes low|medium|high|xhigh. "none" and "minimal" are Responses
# API spellings it rejects outright (instant 400, no retry will help). mirrorbench:616 sends
# effort="none", so without this remap the fallback would miss the exact call site that has
# killed four runs -- measured by stress_fallback.py scenario 3.
_METAGEN_EFFORT_ALIASES = {"none": "low", "minimal": "low"}
_metagen_client = None
_metagen_disabled_reason = None


def _metagen_model(model):
    """AI Gateway registry name -> MetaGen entitlement name; inverse of normalize_model.

    e.g. gpt-5.5 -> gpt-5-5-genai-responses.
    """
    return model.replace(".", "-") + "-genai-responses"


def _get_metagen_client():
    """Cached AsyncOpenAI client, or None when the fallback is not configured.

    The reason is logged once, not per call -- an outage would otherwise print it thousands
    of times.
    """
    global _metagen_client, _metagen_disabled_reason
    if _metagen_client is not None or _metagen_disabled_reason is not None:
        return _metagen_client
    key = os.environ.get("LLAMA_API_KEY")
    if not key:
        _metagen_disabled_reason = "LLAMA_API_KEY is not set"
    else:
        try:
            from openai import AsyncOpenAI

            _metagen_client = AsyncOpenAI(api_key=key, base_url=METAGEN_BASE_URL, max_retries=0)
            print(f"[api.py] MetaGen fallback armed (after {_FALLBACK_AFTER} failed attempts)")
            return _metagen_client
        except Exception as e:  # noqa: BLE001 - never let setup break the primary path
            _metagen_disabled_reason = f"{type(e).__name__}: {e}"
    print(f"[api.py] MetaGen fallback disabled: {_metagen_disabled_reason}")
    return None


async def _metagen_async(payload, timeout):
    """Serve a Responses-shaped payload from MetaGen. Returns the text, or None on failure.

    MetaGen speaks chat-completions, so the Responses fields map back: input -> messages,
    max_output_tokens -> max_completion_tokens, reasoning.effort -> reasoning_effort,
    text.format -> response_format.
    """
    client = _get_metagen_client()
    if client is None:
        return None
    model = _metagen_model(payload["model"])
    kwargs = {
        "model": model,
        "messages": payload["input"],
        "max_completion_tokens": payload.get("max_output_tokens"),
        "timeout": timeout,
    }
    if effort := (payload.get("reasoning") or {}).get("effort"):
        kwargs["reasoning_effort"] = _METAGEN_EFFORT_ALIASES.get(effort, effort)
    if fmt := (payload.get("text") or {}).get("format"):
        kwargs["response_format"] = fmt
    started = time.monotonic()
    try:
        completion = await client.chat.completions.create(**kwargs)
        text = completion.choices[0].message.content or ""
        print(
            f"[api.py] MetaGen served the call in {time.monotonic() - started:.1f}s "
            f"[{len(text):,} chars, model={model}]"
        )
        return text
    except Exception as e:  # noqa: BLE001 - fallback failure must not mask the gateway error
        print(
            f"[api.py] MetaGen fallback failed after {time.monotonic() - started:.1f}s "
            f"[model={model}]: {type(e).__name__}: {str(e)[:140]}"
        )
        return None


# ── Retry budget ──────────────────────────────────────────────────────────────────────
# Every number here is measured, not guessed (stress_fallback.py + job 336216):
#
#   healthy gateway call ......... 25-30s median, 34s max at 32-way concurrency
#   failing gateway attempt ...... exactly 60.2s (its ingress cap) in 390 of 408 failures
#   observed outage length ....... 71+ minutes, ~90% of requests failing
#   MetaGen call ................. ~2s solo, 8s median at 64-way concurrency
#
# The old config (50 retries x 60s + uncapped linear backoff) could block a single call for
# 71 minutes. That is the actual bug: a training step waits on its slowest rollout, and steps
# normally take ~200s. A call that cannot be served in a few minutes should give up and let
# the rollout score zero -- one lost sample out of 512 beats stalling the step.
#
# _MAX_ELAPSED is the real bound; max_retries is now just a backstop. 420s allows the three
# gateway-only attempts (~186s) plus roughly four interleaved rounds after MetaGen joins.
# _MIN_ATTEMPTS is a floor per endpoint, honoured BEFORE the deadline may fire, because a
# wall-clock-only budget silently starved the retry loop: job 337693 gave up after just ~6
# gateway attempts and 2 MetaGen attempts because 420s only buys six 60s timeouts.
# _MAX_ELAPSED is now sized to let both floors complete: 20 gateway attempts at the 60s
# ingress cap is already 1200s, plus MetaGen attempts and backoff.
_MIN_ATTEMPTS = int(os.getenv("API_MIN_ATTEMPTS", "20"))
_MAX_ELAPSED = float(os.getenv("API_MAX_ELAPSED", "2400"))
# Absolute backstop so a pathologically slow endpoint cannot resurrect the 71-minute stall
# that killed 332870 / 334964 / 336216. This one ignores the floors.
_HARD_MAX_ELAPSED = float(os.getenv("API_HARD_MAX_ELAPSED", "3600"))
# Backoff is capped because it now spaces out FALLBACK retries too, not just gateway ones:
# uncapped, attempt 15 waits 15s, which is how one stress run took 128s to do ~2s of work.
# Jitter because hundreds of rollouts fail in lockstep during an outage and would otherwise
# retry in a synchronised thundering herd.
_BACKOFF_CAP = float(os.getenv("API_BACKOFF_CAP", "5"))
# MetaGen answers in single-digit seconds; it never needs the gateway's budget, and a slow
# fallback must not eat the deadline that the gateway attempts still need.
_METAGEN_TIMEOUT = float(os.getenv("API_METAGEN_TIMEOUT", "60"))


def _backoff(attempt):
    return min(attempt + 1, _BACKOFF_CAP) * (0.5 + random.random())


# Cap dumps per process. An outage exhausts many calls at once -- job 336216 had four in one
# 71-minute window, and that scales with the number of in-flight rollouts.
_MAX_REQUEST_DUMPS = 20
_dumps_written = 0


def _dump_failed_request(payload, err, gw_attempts, mg_attempts, elapsed, outcome="gave_up"):
    """Write a request the gateway could not serve, for post-mortem.

    Two outcomes reach here: ``gave_up`` (both endpoints exhausted) and ``rescued_by_metagen``
    (the gateway failed but the fallback answered). The second matters because the call
    SUCCEEDS, so nothing else would record it -- and once the fallback is doing its job that
    becomes the common shape of a gateway problem, silently invisible.

    Sizes alone cannot separate "the gateway is down" from "this request is bad": in job
    336216 the failures ran from 752B to 37KB while an 18.9KB call succeeded in the same
    window. Keeping the full body lets that be settled by looking rather than inferring.
    Diagnostics must never break the caller, so failures here are swallowed.
    """
    global _dumps_written
    if _dumps_written >= _MAX_REQUEST_DUMPS:
        return
    try:
        dump_dir = os.environ.get("API_DUMP_DIR") or os.path.join(os.getcwd(), "api_failures")
        os.makedirs(dump_dir, exist_ok=True)
        _dumps_written += 1
        path = os.path.join(
            dump_dir, f"{outcome}-{os.getpid()}-{_dumps_written}-{int(time.time())}.json"
        )
        with open(path, "w") as fh:
            json.dump(
                {
                    "outcome": outcome,
                    "error": str(err),
                    "gateway_attempts": gw_attempts,
                    "metagen_attempts": mg_attempts,
                    "elapsed_s": round(elapsed, 1),
                    "payload_bytes": len(json.dumps(payload)),
                    "payload": payload,
                },
                fh,
                indent=1,
            )
        print(f"[api.py] wrote {outcome} request to {path}")
    except Exception as e:  # noqa: BLE001 - diagnostics must not mask the original error
        print(f"[api.py] could not dump failed request: {e}")


async def _responses_async(payload, timeout=90, max_retries=50, extract=None):
    """POST a Responses payload to AI Gateway, with retries. Returns the reply text.

    Two in-band recoveries, neither of which burns the backoff sleep:
      * server rejects the ``reasoning`` field  -> strip it and retry, so one bad
        param doesn't fail every judge call;
      * response comes back starved (all budget spent on reasoning, no message)
        -> double max_output_tokens and retry, up to the cap.
    ``extract`` overrides how the successful body is turned into a return value
    (default: the assistant text). Tool-calling callers pass an extractor that
    also surfaces function_call items, so they get the retry, MetaGen-fallback
    and starvation recovery below instead of reimplementing them.
    Raises RuntimeError once retries are exhausted.

    Every attempt that does not return is logged with its wall-clock, the payload size and
    the total elapsed so far. The POST line below fires once per *call*, not per attempt,
    so without this a retry storm is invisible: 50 attempts against a degraded gateway look
    exactly like one fast call, and the run simply goes quiet for an hour before raising.
    That is how jobs 332870 / 334964 died with no warning in the log.
    """
    url = responses_url(payload.get("model"))
    print(f"[api.py] POST {url} with model {payload.get('model', 'unknown')}")
    body = json.dumps(payload)
    last_err = None
    started = time.monotonic()
    # Clamped so short-retry callers still get a fallback: chat_parse drives this with
    # max_retries=2, which would never reach a fixed threshold of 3.
    fallback_after = min(_FALLBACK_AFTER, max(1, max_retries - 1))
    gw_attempts = mg_attempts = 0
    stop_reason = None
    for attempt in range(max_retries):
        elapsed = time.monotonic() - started
        # Each endpoint gets _MIN_ATTEMPTS before wall-clock may end the call. A pure
        # deadline cut job 337693 off after ~6 gateway tries, which is not a real attempt at
        # riding out a degraded window. _HARD_MAX_ELAPSED still bounds the worst case.
        metagen_done = mg_attempts >= _MIN_ATTEMPTS or _metagen_disabled_reason is not None
        floors_met = gw_attempts >= _MIN_ATTEMPTS and metagen_done
        if elapsed > _HARD_MAX_ELAPSED:
            stop_reason = f"hard ceiling {int(_HARD_MAX_ELAPSED)}s reached"
            break
        if floors_met and elapsed > _MAX_ELAPSED:
            stop_reason = f"deadline {int(_MAX_ELAPSED)}s reached"
            break
        if attempt >= fallback_after:
            mg_attempts += 1
            text = await _metagen_async(payload, min(timeout, _METAGEN_TIMEOUT))
            if text is not None:
                # The call succeeds, so nothing downstream records that the gateway could not
                # serve it. Dump the input here or that failure is invisible.
                _dump_failed_request(
                    payload, last_err, gw_attempts, mg_attempts,
                    time.monotonic() - started, "rescued_by_metagen",
                )
                return text
        gw_attempts += 1
        attempt_started = time.monotonic()
        rc, out, err = await _post_async(url, body, timeout)
        took = time.monotonic() - attempt_started

        # Loop vars bound as defaults: the retry branches below reassign ``body``, and this
        # should report the payload that was actually sent (also keeps ruff's B023 quiet).
        def _note(what, detail="", attempt=attempt, took=took, nbytes=len(body)):
            # Detail last and clipped: a gateway error carries the whole model registry, and
            # a storm prints this 50x per call.
            print(
                f"[api.py] attempt {attempt + 1}/{max_retries} {what} after {took:.1f}s "
                f"[{nbytes:,}B payload, {time.monotonic() - started:.0f}s elapsed]"
                + (f": {detail[:160]}" if detail else "")
            )

        if rc != 0:
            last_err = RuntimeError(f"curl failed (exit {rc}): {err.strip()}")
        else:
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                data = None
                last_err = RuntimeError(f"non-JSON response: {out[:200]!r}")
            if isinstance(data, dict):
                if data.get("error"):
                    if "reasoning" in payload and _mentions_reasoning(data["error"]):
                        payload = {k: v for k, v in payload.items() if k != "reasoning"}
                        body = json.dumps(payload)
                        _note("rejected the reasoning field, retrying without it")
                        continue
                    last_err = RuntimeError(f"AI Gateway error: {data['error']}")
                elif _starved(data):
                    grown = min(payload["max_output_tokens"] * 2, _MAX_OUTPUT_TOKENS_CAP)
                    last_err = RuntimeError(
                        f"response starved by reasoning at max_output_tokens="
                        f"{payload['max_output_tokens']}"
                    )
                    if grown > payload["max_output_tokens"]:
                        payload = {**payload, "max_output_tokens": grown}
                        body = json.dumps(payload)
                        _note(f"came back starved, retrying at max_output_tokens={grown}")
                        continue
                elif "output" in data:
                    if attempt:
                        _note("succeeded")
                    return (extract or _responses_text)(data)
                else:
                    last_err = RuntimeError(f"AI Gateway error: {data}")
        _note("failed", str(last_err))
        if attempt < max_retries - 1:
            await asyncio.sleep(_backoff(attempt))
    elapsed = time.monotonic() - started
    print(
        f"[api.py] giving up after {elapsed:.0f}s "
        f"({gw_attempts} gateway + {mg_attempts} metagen attempts; "
        f"{stop_reason or f'{max_retries} attempts exhausted'}): {str(last_err)[:160]}"
    )
    _dump_failed_request(payload, last_err, gw_attempts, mg_attempts, elapsed)
    raise last_err or RuntimeError("responses completion failed after retries")


# ── Gemini transport (Vertex :generateContent) ────────────────────────────────────────
def _gemini_url(model, method="generateContent"):
    return (
        f"{GEMINI_HOST}/v1/projects/{GEMINI_PROJECT}/locations/{GEMINI_LOCATION}"
        f"/publishers/google/models/{model}:{method}"
    )


def _to_gemini_payload(messages, max_tokens, response_format=None, reasoning_effort=None):
    """Build a Vertex :generateContent body from chat-completions-style arguments.

    Field mapping: messages -> contents (role 'assistant' becomes 'model', and the
    'system' role is lifted out to the top-level systemInstruction, which is where
    Vertex wants it), max_tokens -> generationConfig.maxOutputTokens,
    response_format -> generationConfig.responseMimeType.

    Consecutive same-role turns are merged into one entry's parts, and empty
    contents are dropped -- Vertex rejects a parts list with no text.
    """
    contents, system_parts = [], []
    for m in messages:
        text = m.get("content") or ""
        if m.get("role") == "system":
            if text:
                system_parts.append({"text": text})
            continue
        if not text:
            continue
        role = "model" if m.get("role") == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": text})
        else:
            contents.append({"role": role, "parts": [{"text": text}]})

    # Gemini bills thinking tokens against maxOutputTokens exactly as the Responses
    # API bills reasoning, so the same headroom applies: asking for 512 on a 2.5-pro
    # JSON call returns a truncated '{"city": "Tokyo' with finishReason MAX_TOKENS.
    effort = normalize_reasoning_effort(reasoning_effort)
    budget = max_tokens if effort == "none" else max_tokens + _REASONING_HEADROOM
    config = {"maxOutputTokens": min(budget, _MAX_OUTPUT_TOKENS_CAP)}
    if effort == "none":
        # Only flash-class models accept this; pro answers 400 "The model does not
        # support setting thinking_budget to 0", which _gemini_async strips and retries.
        config["thinkingConfig"] = {"thinkingBudget": 0}
    if (response_format or {}).get("type") == "json_object":
        config["responseMimeType"] = "application/json"
    payload = {"contents": contents, "generationConfig": config}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


def _gemini_text(data):
    """Extract the reply text from a :generateContent body.

    Thinking produces no text part, so a candidate can legitimately carry none;
    return the first candidate that has any.
    """
    for cand in data.get("candidates", []):
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        if text:
            return text
    return ""


def _gemini_starved(data):
    """True if thinking consumed the whole budget and no text came back."""
    cands = data.get("candidates") or []
    return bool(cands) and cands[0].get("finishReason") == "MAX_TOKENS" and not _gemini_text(data)


def _mentions_thinking(data) -> bool:
    return "thinking" in json.dumps(data).lower()


async def _gemini_async(model, payload, timeout=90, max_retries=50):
    """POST a :generateContent payload, with retries. Returns the reply text.

    Mirrors _responses_async's two in-band recoveries -- strip a rejected
    thinkingConfig, grow a starved budget -- but has no MetaGen fallback, which
    serves no Gemini model.
    """
    url = _gemini_url(model)
    print(f"[api.py] POST {GEMINI_HOST} :generateContent with model {model}")
    body = json.dumps(payload)
    last_err = None
    started = time.monotonic()
    for attempt in range(max_retries):
        if time.monotonic() - started > _HARD_MAX_ELAPSED:
            break
        rc, out, err = await _post_async(url, body, timeout)
        if rc != 0:
            last_err = RuntimeError(f"curl failed (exit {rc}): {err.strip()}")
        else:
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                data = None
                last_err = RuntimeError(f"non-JSON response: {out[:200]!r}")
            if isinstance(data, dict):
                if data.get("error"):
                    cfg = payload.get("generationConfig", {})
                    if "thinkingConfig" in cfg and _mentions_thinking(data["error"]):
                        cfg = {k: v for k, v in cfg.items() if k != "thinkingConfig"}
                        payload = {**payload, "generationConfig": cfg}
                        body = json.dumps(payload)
                        continue
                    last_err = RuntimeError(f"AI Gateway error: {data['error']}")
                elif _gemini_starved(data):
                    cfg = payload["generationConfig"]
                    grown = min(cfg["maxOutputTokens"] * 2, _MAX_OUTPUT_TOKENS_CAP)
                    last_err = RuntimeError(
                        f"response starved by thinking at maxOutputTokens="
                        f"{cfg['maxOutputTokens']}"
                    )
                    if grown > cfg["maxOutputTokens"]:
                        payload = {**payload, "generationConfig": {**cfg, "maxOutputTokens": grown}}
                        body = json.dumps(payload)
                        continue
                elif "candidates" in data:
                    return _gemini_text(data)
                else:
                    last_err = RuntimeError(f"AI Gateway error: {data}")
        if attempt < max_retries - 1:
            await asyncio.sleep(_backoff(attempt))
    raise last_err or RuntimeError("gemini completion failed after retries")


def chat(prompt, model=DEFAULT_MODEL, system=None, max_tokens=2048, timeout=120):
    """Send a completion through AI Gateway and return the reply text.

    Routes on the model name: gemini-* to the Vertex gateway, everything else to
    the Responses route (where gpt-5.5 is the only deployment). For Claude, use
    chat_messages(). See AVAILABLE_MODELS / GEMINI_MODELS.
    """
    if is_gemini_model(model):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        result = subprocess.run(
            _curl_base(_gemini_url(model), timeout),
            input=json.dumps(_to_gemini_payload(messages, max_tokens)),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr.strip()}")
        data = json.loads(result.stdout)
        if data.get("error") or "candidates" not in data:
            raise RuntimeError(f"AI Gateway error: {data.get('error') or data}")
        return _gemini_text(data)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps(_to_responses_payload(messages, model, max_tokens))

    result = subprocess.run(
        _curl_base(responses_url(model), timeout),
        input=body,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if data.get("error") or "output" not in data:
        raise RuntimeError(f"AI Gateway error: {data.get('error') or data}")
    return _responses_text(data)


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


async def chat_async(
    messages,
    model=DEFAULT_MODEL,
    system=None,
    max_tokens=2048,
    timeout=90,
    max_retries=50,
    response_format=None,
    reasoning_effort=None,
    seed=None,
):
    """Async completion returning the reply text. The AI Gateway/curl equivalent
    of ``client.chat.completions.create(...).choices[0].message.content``.

    ``messages`` may be a prompt string or an OpenAI-style messages list.
    ``seed`` is accepted and ignored -- the Responses API has no such parameter
    and rejects it outright; the argument is kept so call sites don't have to
    change. Raises on failure after retries (callers that want a soft failure
    should catch and substitute "").
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + messages

    if is_gemini_model(model):
        payload = _to_gemini_payload(messages, max_tokens, response_format, reasoning_effort)
        return await _gemini_async(model, payload, timeout, max_retries) or ""

    if is_messages_model(model):
        # Anthropic takes the system prompt as a top-level field and rejects system
        # turns inside the list, so lift them out and coerce the rest into strict
        # alternation -- the same preparation agents/utils.py does before it calls
        # chat_messages_async directly. Without this branch a Claude name reaching
        # chat_async would be posted to Azure and come back DeploymentNotFound,
        # which says nothing about the actual mistake.
        sys_text = (
            "\n".join(m["content"] for m in messages if m["role"] == "system" and m["content"])
            or None
        )
        conv = merge_alternating([m for m in messages if m["role"] != "system"])
        return await chat_messages_async(
            conv, model=model, system=sys_text, max_tokens=max_tokens,
            timeout=timeout, max_retries=max_retries,
        ) or ""

    payload = _to_responses_payload(
        messages, model, max_tokens, response_format, reasoning_effort
    )
    return await _responses_async(payload, timeout, max_retries) or ""


async def chat_tools_async(
    messages, model=DEFAULT_MODEL, tools=None, tool_choice="required",
    max_tokens=2048, timeout=90, max_retries=8, reasoning_effort=None,
):
    """Completion that may return TOOL CALLS -- for driving a hosted model as the
    agent rather than the simulator.

    Returns ``(text, tool_calls)`` with tool_calls in OpenAI chat-completions
    shape, so a shim can put them straight into a `message` object.

    Responses-only on purpose. Verified 2026-09-24: the Azure /responses route
    returns output=['reasoning','function_call'] for gpt-5.6-luna and -sol with
    the gyms' interact_with_env schema, while atlas-cloud answers
    400 "invalid request params" for the same model and tools. Anthropic and
    Vertex have their own tool formats and are not wired up here.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    payload = _to_responses_payload(
        messages, model, max_tokens, None, reasoning_effort,
        tools=tools, tool_choice=tool_choice,
    )
    return await _responses_async(
        payload, timeout, max_retries,
        extract=lambda d: (_responses_text(d), _responses_tool_calls(d)),
    )


async def chat_parse(
    messages,
    text_format,
    model=DEFAULT_MODEL,
    max_tokens=2048,
    timeout=90,
    max_retries=50,
    reasoning_effort=None,
):
    """Structured-output completion: the equivalent of the OpenAI SDK's
    ``client.responses.parse(...)``.

    ``text_format`` is a pydantic ``BaseModel`` subclass. The gateway ignores
    OpenAI strict ``json_schema`` outputs, so this uses JSON mode
    (``text.format={"type":"json_object"}``) with the schema embedded in the
    prompt, then validates client-side.

    Returns ``model_dump(by_alias=True)`` (a dict) on success. Retries
    ``max_retries`` times, then raises the last error.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    schema = text_format.model_json_schema()
    messages = messages + [
        {
            "role": "system",
            "content": (
                "Respond with ONLY a single JSON object that conforms to this "
                "JSON Schema. No markdown code fences, no prose, no explanation.\n"
                f"JSON Schema:\n{json.dumps(schema)}"
            ),
        }
    ]

    gemini = is_gemini_model(model)
    if gemini:
        payload = _to_gemini_payload(
            messages, max_tokens, {"type": "json_object"}, reasoning_effort
        )
    else:
        payload = _to_responses_payload(
            messages, model, max_tokens, {"type": "json_object"}, reasoning_effort
        )

    last_err = None
    for attempt in range(max_retries):
        try:
            # Inner max_retries=2 leaves room for the reasoning strip-retry inside
            # _responses_async; validation retries are this outer loop.
            if gemini:
                content = await _gemini_async(model, payload, timeout, max_retries=2)
            else:
                content = await _responses_async(payload, timeout, max_retries=2)
            content = _JSON_FENCE.sub("", content or "").strip()
            if not content:
                raise ValueError("empty content")
            obj = text_format.model_validate_json(content)
            return obj.model_dump(by_alias=True)
        except Exception as e:
            last_err = e
            print(f"[chat_parse] attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
    raise last_err or RuntimeError("chat_parse failed after retries")


def _messages_text(data):
    """Extract concatenated text from an Anthropic /v1/messages response.

    The reply is a list of content blocks; we join the text of every 'text'
    block (thinking/tool blocks, if any, are skipped) so the return type matches
    chat()/chat_async() -- a plain string.
    """
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(parts)


def chat_messages(prompt, model=DEFAULT_MESSAGES_MODEL, system=None, max_tokens=2048, timeout=120):
    """Send a message through the Anthropic-native /v1/messages route and
    return the reply text. Use this for every Claude model; see
    ANTHROPIC_MESSAGES_MODELS for the ones actually deployed.

    Unlike chat(), the system prompt is a top-level field (not a message) and
    max_tokens is required by the API.
    """
    body = {
        "model": normalize_model(model),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    result = subprocess.run(
        _curl_base(MESSAGES_URL, timeout, _MESSAGES_HEADERS),
        input=json.dumps(body),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed (exit {result.returncode}): {result.stderr.strip()}")
    data = json.loads(result.stdout)
    if "content" not in data:
        raise RuntimeError(f"AI Gateway error: {data}")
    return _messages_text(data)


async def chat_messages_async(
    messages,
    model=DEFAULT_MESSAGES_MODEL,
    system=None,
    max_tokens=2048,
    timeout=120,
    max_retries=50,
):
    """Async /v1/messages completion returning the reply text -- the Anthropic-
    native counterpart of chat_async(), for the Claude models.

    ``messages`` may be a prompt string or an Anthropic-style messages list
    (roles limited to 'user'/'assistant'; system goes in the ``system`` arg, not
    the list). Retries with linear backoff, mirroring _responses_async.
    Raises on failure after retries.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    model = normalize_model(model)
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        payload["system"] = system
    print(f"[api.py] POST {MESSAGES_URL} with model {model}")
    body = json.dumps(payload)

    last_err = None
    for attempt in range(max_retries):
        rc, out, err = await _post_async(MESSAGES_URL, body, timeout, _MESSAGES_HEADERS)
        if rc != 0:
            last_err = RuntimeError(f"curl failed (exit {rc}): {err.strip()}")
        else:
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                data = None
                last_err = RuntimeError(f"non-JSON response: {out[:200]!r}")
            if isinstance(data, dict) and "content" in data:
                return _messages_text(data)
            if isinstance(data, dict):
                last_err = RuntimeError(f"AI Gateway error: {data}")
        if attempt < max_retries - 1:
            await asyncio.sleep(1 * (attempt + 1))
    raise last_err or RuntimeError("chat_messages completion failed after retries")


if __name__ == "__main__":
    print(chat_messages("Tell me a joke", model="claude-opus-5"))
