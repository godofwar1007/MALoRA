from baseloader import BaseDatasetLoader  # adjust to your actual import path
'''
this has been checked
this dataset will have a high context_overflow rate
bruh 
:C
'''


class MagicoderOSSLoader(BaseDatasetLoader):
    """
    Loader for ise-uiuc/Magicoder-OSS-Instruct-75K

    Schema:
        problem  : str — NL coding problem, often includes a partial snippet
        solution : str — fenced code block (```lang...```) + prose explanation

    Format: single-turn
        user      <- problem
        assistant <- solution (verbatim — fenced block teaches the model the language)

    Skipped (via base class):
        - format_rejected  : problem or solution missing / empty  (_format returns None)
        - context_overflow : tokenized length >= context_length   (overflow_flag)
        - all_labels_masked: no assistant boundary found          (all labels == -100)
    """

    HF_ID  = "ise-uiuc/Magicoder-OSS-Instruct-75K"
    SUBSET = None
    SPLIT  = "train"

    def _format(self, example: dict) -> list[dict] | None:
        problem  = (example.get("problem")  or "").strip()
        solution = (example.get("solution") or "").strip()

        if not problem or not solution:
            return None

        return [
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": solution},
        ]