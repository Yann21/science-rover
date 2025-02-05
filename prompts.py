# %%
import re
from utils import Paper, Message


def batch_relevance_score_prompt(original_paper: Paper, comparant_papers: list[Paper]) -> list[Message]:
  # create a variable abstracts which contains all the abstracts except the first one
  # and that is introduced by paper #1: ... paper #2: ... etc.
  messages = [
    Message(
      "system",
      ("Given the following abstracts, please score their relevance to the original paper."
       "The score should be an integer from 1 to 9. So an example reply could be '4,8,9,1,2,5,4,6,6,9'."
       f"Original paper: **{original_paper.title}**\n{original_paper.abstract}")
    ),
    Message(
      "user",
      "\n\n".join([
        f"Paper #{i + 1}:\n**{paper.title}**\n{paper.abstract}"
        for i, paper in enumerate(comparant_papers[1:])
      ])
    )
  ]

  return messages


def contribution_prompt(chain: list[Paper]) -> list[Message]:
  messages = [
    Message(
      "system",
      "Given the following papers that are known to be related, please provide an interdisciplinary contribution to the field that could combine their ideas."
    ),
    Message(
      "user",
      "\n\n".join([
        f"**{paper.title}**\n{paper.abstract}"
        for paper in chain
      ])
    )
  ]

  return messages


re_batch_relevance_extraction = re.compile(r"(\d+),?")
