from baseloader import BaseDatasetLoader
'''
THIS FILE HAS BEEN TESTED
'''

class CodeContestsLoader(BaseDatasetLoader):

    HF_ID  = "ad6398/Deepmind-CodeContest-Unrolled"
    SUBSET = None
    SPLIT  = "train"

    ALLOWED_LANGS = {"cpp", "java", "python3"}

    def _format(self, example):
        if example["solution_type"] != "CORRECT":
            return None
        if example["programming_language"] not in self.ALLOWED_LANGS:
            return None
        return [
            {"role": "user", "content": example["description"].strip()},
            {"role": "assistant", "content": (
                f"# Language: {example['programming_language']}\n\n"
                f"{example['solution'].strip()}"
            )},
        ]