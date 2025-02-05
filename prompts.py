# %%
import re


def batch_relevance_score_prompt(abstracts: list[str]) -> str:
  # create a variable abstracts which contains all the abstracts except the first one
  # and that is introduced by abstract #1: ... abstract #2: ... etc.
  abstracts = [
    f"abstract #{i + 1}:\n{abstract}" for i, abstract in enumerate(abstracts[1:])
  ]
  abstracts = "\n\n".join(abstracts)

  return f"""Given the following abstracts, please score their relevance to the starting paper.
The score should be an integer from 1 to 9. So an example reply could be "9,8,7,6,5,4,3,2,1".


Original paper: {abstracts[0]}

{abstracts}
"""


def contribution_prompt(abstracts: str) -> str:
  return f"""Given the following abstracts that are known to be related, please provide 
an interdisciplinary contribution to the field that could combine their ideas.

{abstracts}"""


re_batch_relevance_extraction = re.compile(r"(\d+),?")
