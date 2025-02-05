# %%
import re
from utils import Paper, Message


def batch_relevance_score_prompt(original_paper: Paper, comparant_papers: list[Paper]) -> list[Message]:
  # create a variable abstracts which contains all the abstracts except the first one
  # and that is introduced by paper #1: ... paper #2: ... etc.
  messages = [
    Message(
      "system",
      ("Given the following abstracts, please score their relevance to the original paper.\n"
       "The score should be an integer from 1 to 9. An example reply should look like:\n"
       "#1: 7\n#2: 5\n#3: 8\n#4: 6\n#5: 9\netc.\n\n"
       f"Original paper: **{original_paper.title}**\n{original_paper.abstract}")
    ),
    Message(
      "user",
      "\n\n".join([
        f"Paper #{i + 1}:\n**{paper.title}**\n{paper.abstract}"
        for i, paper in enumerate(comparant_papers)
      ])
    )
  ]

  return messages


re_batch_relevance_extraction = re.compile(r": (\d+)")


def contribution_prompt(chain: list[Paper]) -> list[Message]:
  messages = [
    Message(
      "system",
      ("Given the following papers that are known to be related, "
       "please provide an interdisciplinary contribution to the field that could combine their ideas.")
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


def auto_criticism_prompt(contribution: str) -> list[Message]:
  messages = [
    Message(
      "system",
      ("Analyze this idea: Evaluate its feasibility, innovation, scientific relevance, and applicability, "
       "and provide a final rating from 1 to 9 with detailed reasoning. "
       "The rating should be the average grade from 1 to 9 on a separate newline rounded to the nearest integer like this:\n"
       "7")
    ),
    Message(
      "user",
      contribution
    )
  ]

  return messages


def extract_auto_criticism_score(reply: str) -> int:
  return int(re.findall(r"\d+", reply)[-1])
