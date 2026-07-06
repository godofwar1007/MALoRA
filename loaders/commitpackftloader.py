from baseloader import BaseDatasetLoader  # adjust to your actual import path
'''
this has been checked
this dataset will have a high context_overflow rate
bruh 
:C
'''
class CodeFeedbackLoader(BaseDatasetLoader):
    HF_ID = "m-a-p/CodeFeedback-Filtered-Instruction"
    SUBSET = None
    SPLIT = "train"

    def _format(self, example) -> list[dict] | None:
        query = (example.get("query") or "").strip()
        answer = (example.get("answer") or "").strip()

        if not query or not answer:
            return None

        return [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ]