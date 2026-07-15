import json
from baseloader import BaseDatasetLoader


class OpenCodeInstructLoader(BaseDatasetLoader):


    HF_ID  = "nvidia/OpenCodeInstruct"
    SUBSET = None
    SPLIT  = "train"

    MIN_TEST_SCORE = 0.8   # fraction of unit tests passed (0.0 - 1.0)
    MIN_LLM_SCORE  = 4.5   # average of 3 criteria scored 1-5 (paper uses 5.0)
                            # lower to 4.5 if you need more samples

    def _parse_test_score(self, score_str) -> float:
        """
        average_test_score is a STRING like '0.9', '1', '0.8'.
        Returns float 0.0-1.0, or -1.0 on failure (treated as unknown → pass filter).
        """
        if not score_str:
            return -1.0
        try:
            return float(score_str)
        except (ValueError, TypeError):
            return -1.0

    def _parse_llm_judgement(self, judgement_str) -> float:
       
        if not judgement_str:
            return -1.0
        try:
            j = json.loads(judgement_str)
            scores = [
                v["score"]
                for v in j.values()
                if isinstance(v, dict) and "score" in v
            ]
            if not scores:
                return -1.0
            return sum(scores) / len(scores)
        except (json.JSONDecodeError, TypeError, KeyError):
            return -1.0

    def _format(self, example: dict) -> list[dict] | None:
        instruction = (example.get("input")  or "").strip()
        response    = (example.get("output") or "").strip()

        if not instruction or not response:
            return None

        # ── test score filter ─────────────────────────────────────────
        test_score = self._parse_test_score(example.get("average_test_score", ""))
        if test_score >= 0.0 and test_score < self.MIN_TEST_SCORE:
            return None

        # ── llm judgement filter ──────────────────────────────────────
        llm_score = self._parse_llm_judgement(example.get("llm_judgement", ""))
        if llm_score >= 0.0 and llm_score < self.MIN_LLM_SCORE:
            return None

        return [
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": response},
        ]