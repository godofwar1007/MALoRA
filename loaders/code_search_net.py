import re
from baseloader import BaseDatasetLoader

SYSTEM_PROMPT = (
    "You are an expert software engineer. "
    "Write a concise, accurate docstring for the given function. "
    "Output only the docstring text — no code, no explanation, no surrounding quotes."
)


def _strip_python_docstring(code: str) -> str | None:
    pattern = re.compile(
        r'(def\s+\w+[^:]*:[ \t]*\n)'
        r'([ \t]*)("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
        re.MULTILINE,
    )

    stripped, n_subs = pattern.subn(r'\1', code, count=1)

    if n_subs == 0:
        return None

    return stripped.strip()


def _strip_javadoc(code: str) -> str | None:
    stripped, n_subs = re.subn(
        r"/\*\*[\s\S]*?\*/",
        "",
        code,
        count=1,
    )

    if n_subs == 0:
        return None

    return stripped.strip()


class CodeSearchNetLoader(BaseDatasetLoader):
    HF_ID  = "claudios/code_search_net"
    SUBSET = "all"
    SPLIT  = "train"

    LANGUAGES = {"python", "java", "javascript"}

    def _format(self, example: dict) -> list[dict] | None:
        language = (example.get("language") or "").lower().strip()

        if language not in self.LANGUAGES:
            return None

        func_code = (example.get("func_code_string") or "").strip()
        docstring = (example.get("func_documentation_string") or "").strip()

        if not func_code or not docstring:
            return None

        if language == "python":
            stripped_code = _strip_python_docstring(func_code)
        else:
            stripped_code = _strip_javadoc(func_code)

        if stripped_code is None:
            return None

        user_content = (
            f"Write a docstring for the following {language} function:\n\n"
            f"```{language}\n{stripped_code}\n```"
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": docstring},
        ]