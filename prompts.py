# %%
import re


def batch_relevance_score_prompt(abstracts: list[str]) -> str:
  # create a variable abstracts which contains all the abstracts except the first one
  # and that is introduced by abstract #1: ... abstract #2: ... etc.
  comparant_abstracts = [
    f"abstract #{i + 1}:\n{abstract}" for i, abstract in enumerate(abstracts[1:])
  ]
  comparant_abstracts = "\n\n".join(comparant_abstracts)

  return f"""Given the following abstracts, please score their relevance to the starting paper.
The score should be an integer from 1 to 9. So an example reply could be "9,8,7,6,5,4,3,2,1".


Original paper: {abstracts[0]}

{comparant_abstracts}
"""


re_batch_relevance_extraction = re.compile(r"(\d+),?")


def contribution_prompt(abstracts: str) -> str:
  return f"""Given the following abstracts that are known to be related, please provide 
an interdisciplinary contribution to the field that could combine their ideas.

{abstracts}"""


def auto_criticism_prompt(contribution: str) -> str:
  return f"""Analyze this idea: Evaluate its feasibility, innovation, scientific relevance, and applicability, and provide a final rating from 1 to 9 with detailed reasoning. The rating should be the average grade from 1 to 9 on a separate newline rounded to the nearest integer like this:
7


{contribution}
"""


def extract_auto_criticism_score(reply: str) -> int:
  return int(reply[-1])
