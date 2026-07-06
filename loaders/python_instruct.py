from baseloader import BaseDatasetLoader
'''
THIS FILE HAS BEEN TESTED
'''
class PythonInstructionsLoader(BaseDatasetLoader):

    HF_ID  = "iamtarun/python_code_instructions_18k_alpaca"
    SUBSET = None
    SPLIT  = "train"

    def _format(self, example):
        prompt = example["prompt"]
        # Strip the output section that's baked into the prompt
        cutoff = prompt.find("### Output:")
        if cutoff != -1:
            prompt = prompt[:cutoff].strip()
        return [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": example["output"].strip()},
        ]