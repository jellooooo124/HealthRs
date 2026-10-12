"""
LLM-simulated user (deepseek-v4.1-flash, temperature 0). The first message is scripted (identical across
configurations); later messages react to the assistant according to the persona's hidden ground truth.
"""
import json
import re

from agent.models.cloud_model import CloudChatModel

COND_TEXT = {"diabetes": "type 2 diabetes", "hypertension": "high blood pressure", "hyperuricemia": "gout / high uric acid"}
ALG_TEXT = {"tree_nut": "tree nuts", "milk": "milk/dairy"}

SIM_PROMPT = """You are role-playing a real person chatting with a recipe assistant app. Stay in character.

Who you are:
- Health: {conditions}. Food allergies: {allergies}.
- {health_disclosure}
- Taste: you like {taste}.{liked}
- You dislike {dislike} (you have not told the assistant yet).
- You have at most {time_limit} minutes to cook.
- Today you want {meal}.

How to reply:
- Write 1-2 short, natural sentences, like a real user typing. Never mention that you are simulated.
- If the assistant asks about your health or allergies, answer truthfully and briefly.
- Read the recommendations. You are SATISFIED if at least one recipe fits what you want today, your time
  limit and your taste, and does not contain {dislike} or anything you are allergic to.
- If you are satisfied, say which recipe you will cook and thank the assistant (set satisfied=true, end=true).
  Picking a recipe to cook ALWAYS means satisfied=true and end=true.
- If not, give concrete feedback (e.g. a recipe has {dislike}, takes too long, is not what you wanted) and ask for
  alternatives. If this is your last message, set end=true.
This is your message number {turn} of at most {max_turns}.

Return ONLY a JSON object, deciding first and writing the message last:
{{"satisfied": true|false, "end": true|false, "chosen_recipe": "name or null", "message": "..."}}"""


def _fmt_list(items, mapping):
    return ", ".join(mapping.get(x, x.replace("_", " ")) for x in items) or "none"


class SimulatedUser:
    def __init__(self, scenario: dict, max_turns: int):
        self.s, self.max_turns = scenario, max_turns
        self.llm = CloudChatModel("user_sim", temperature=0.0, max_tokens=1500).llm

    def opener(self) -> str:
        return self.s["opener"]

    def system_prompt(self, turn: int) -> str:
        s = self.s
        disclosure = ("The assistant already knows your health conditions and allergies from your profile."
                      if s["kind"] == "warm" else
                      "This is your first time using the app; the assistant does not know your health yet.")
        liked = f" Recently you enjoyed: {'; '.join(s['liked_recipes'])}." if s["liked_recipes"] else ""
        return SIM_PROMPT.format(conditions=_fmt_list(s["conditions"], COND_TEXT), allergies=_fmt_list(s["allergies"], ALG_TEXT),
                                 health_disclosure=disclosure, taste=", ".join(s["taste"]) or "simple home cooking",
                                 liked=liked, dislike=s["dislike"], time_limit=s["time_limit"], meal=s["meal"],
                                 turn=turn, max_turns=self.max_turns)

    async def reply(self, transcript: list[dict], turn: int, callbacks: list | None = None) -> dict:
        """turn = number of the user message being written; turn > max_turns means: final verdict only."""
        prompt = self.system_prompt(min(turn, self.max_turns))
        if turn > self.max_turns:
            prompt += ("\nThe conversation is now over. Only decide whether the assistant's LAST answer satisfies you; "
                       "'message' is a one-line closing remark and end=true.")
        # the simulator plays the user: the assistant's messages are its "user" input
        msgs = [("system", prompt)]
        for m in transcript:
            msgs.append(("assistant" if m["role"] == "user" else "user", m["content"]))
        out = await self.llm.ainvoke(msgs, config={"callbacks": callbacks or []})
        text = out.content if isinstance(out.content, str) else str(out.content)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        chosen = data.get("chosen_recipe")
        chosen = None if str(chosen).strip().lower() in ("", "none", "null") else chosen
        satisfied = bool(data.get("satisfied", False)) or chosen is not None
        return {"message": str(data.get("message") or text.strip()[:400]), "satisfied": satisfied,
                "end": bool(data.get("end", False)) or turn > self.max_turns, "chosen_recipe": chosen}
