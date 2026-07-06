
from baseloader import BaseDatasetLoader
'''
this file has been checked
'''

class SqlCreateContextLoaderBi(BaseDatasetLoader):
    """
    b-mc2/sql-create-context — bidirectional variant.
    Each raw example yields 2 training samples:
      [A] context + question  -> SQL query      (NL-to-SQL)
      [B] context + SQL query -> question       (SQL-to-NL)
    """

    HF_ID  = "b-mc2/sql-create-context"
    SUBSET = None
    SPLIT  = "train"

    _SYS_NL_TO_SQL = (
        "You are an expert SQL assistant. "
        "Given a database schema and a natural language question, "
        "write a correct SQL query that answers the question. "
        "Return only the SQL query with no explanation."
    )

    _SYS_SQL_TO_NL = (
        "You are an expert SQL assistant. "
        "Given a database schema and a SQL query, "
        "write the natural language question that the query is answering. "
        "Return only the question with no explanation."
    )

    def _format(self, example) -> list[list[dict]] | None:
        answer   = (example.get("answer")   or "").strip()
        question = (example.get("question") or "").strip()
        context  = (example.get("context")  or "").strip()

        if not answer or not question or not context:
            return None

        nl_to_sql = [
            {"role": "system",    "content": self._SYS_NL_TO_SQL},
            {"role": "user",      "content": f"{context}\n\n{question}"},
            {"role": "assistant", "content": answer},
        ]

        sql_to_nl = [
            {"role": "system",    "content": self._SYS_SQL_TO_NL},
            {"role": "user",      "content": f"{context}\n\n{answer}"},
            {"role": "assistant", "content": question},
        ]

        return [nl_to_sql, sql_to_nl]