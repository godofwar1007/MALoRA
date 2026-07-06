from datasets import load_dataset
from baseloader import BaseDatasetLoader
'''
this dataset has been tested
it is very good 
1 training sample here gives 10 samples for models 
very good but the context window issue still stays
'''

# Display names for prompt strings
LANG_DISPLAY = {
    "java":       "Java",
    "c++":        "C++",
    "python":     "Python",
    "javascript": "JavaScript",
}

LANGUAGES = list(LANG_DISPLAY.keys())


class LeetCodeLoader(BaseDatasetLoader):
    HF_ID  = "greengerong/leetcode"
    SUBSET = None
    SPLIT  = "train"

    def _format(self, example) -> list[list[dict]] | None:
        content = (example.get("content") or "").strip()
        if not content:
            return None

        # Collect non-empty language solutions
        solutions: dict[str, str] = {}
        for lang in LANGUAGES:
            code = (example.get(lang) or "").strip()
            if code:
                solutions[lang] = code

        # Need at least one solution to do anything useful
        if not solutions:
            return None

        samples: list[list[dict]] = []

        # ── Q → lang (one sample per available language) ─────────────
        for lang, code in solutions.items():
            display = LANG_DISPLAY[lang]
            samples.append([
                {
                    "role": "user",
                    "content": (
                        f"Solve the following coding problem in {display}:\n\n"
                        f"{content}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": code,
                },
            ])

        # ── A → B translation (all ordered pairs, no problem statement) ─
        lang_list = list(solutions.keys())
        for i, src_lang in enumerate(lang_list):
            for j, tgt_lang in enumerate(lang_list):
                if i == j:
                    continue
                src_display = LANG_DISPLAY[src_lang]
                tgt_display = LANG_DISPLAY[tgt_lang]
                samples.append([
                    {
                        "role": "user",
                        "content": (
                            f"Convert the following {src_display} solution "
                            f"to {tgt_display}:\n\n"
                            f"{solutions[src_lang]}"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": solutions[tgt_lang],
                    },
                ])

        return samples if samples else None