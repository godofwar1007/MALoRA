from baseloader import BaseDatasetLoader
'''
THIS FILE HAS BEEN TESTED
'''

class CodeAlpacaLoader(BaseDatasetLoader):

    HF_ID  = "sahil2801/CodeAlpaca-20k"
    SUBSET = None
    SPLIT  = "train"

    def _format(self, example):
        user_content = example["instruction"].strip()
        if example.get("input", "").strip():
            user_content += f"\n\n{example['input'].strip()}"
        return [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example["output"].strip()},
        ]