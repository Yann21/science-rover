import re


def batch_relevance_score_template(abstracts: list[str]) -> str:
  return f"Given the following abstracts, please score their relevance to the starting paper:\n\n{abstracts}\n\n"


re_batch_relevance_extraction = re.compile(r"(\d,?)+")
