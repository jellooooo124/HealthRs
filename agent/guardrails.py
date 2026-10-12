"""
Input/output guardrails:
  1. LLM-as-judge input moderation (switch: guardrails.llm_check — on for the demo, off in evaluation)
  2. PII redaction (emails, phone numbers, ID and card numbers) on input and output
"""
import re

from src.config import get_agent_config, load_config
from src.utils.logger import get_logger

logger = get_logger("healthrec.guardrails")
_judge = None

_PII = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[email redacted]"),
    (re.compile(r"\+\d{1,3}[\s\-]?\(?\d{1,5}\)?(?:[\s\-]?\d{3,5}){1,3}(?=\s|$|[^\d])"), "[phone redacted]"),
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[phone redacted]"),
    (re.compile(r"\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dX]\b", re.I), "[ID redacted]"),
    (re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"), "[card number redacted]"),
]


def redact_pii(text: str) -> str:
    for pattern, repl in _PII:
        text = pattern.sub(repl, text)
    return text


async def check_input(text: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). Fails open if the judge is unavailable."""
    global _judge
    if not get_agent_config().get("guardrails", {}).get("llm_check", True):
        return True, ""
    try:
        if _judge is None:
            from agent.models.cloud_model import CloudChatModel
            _judge = CloudChatModel("agent", temperature=0.0).llm_safe
        out = await _judge.ainvoke([("system", load_config("prompts")["safety_judge"]), ("user", text[:500])])
        verdict = out.content.strip()
        if verdict.upper().startswith("UNSAFE"):
            reason = verdict.split(":", 1)[-1].strip() if ":" in verdict else "inappropriate content"
            logger.warning(f"[guardrails] blocked input: {reason}")
            return False, reason
    except Exception as e:
        logger.warning(f"[guardrails] judge failed, letting input through: {e}")
    return True, ""
