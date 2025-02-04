# %%
from typing import Dict, List
from dataclasses import dataclass
from random import sample
from prompts import batch_relevance_score_template, re_batch_relevance_extraction
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


# %%


@dataclass
class Paper:
  abstract: str
  title: str


RelevanceScore = int


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_gpt_4o(prompt):
  chat_completion = client.chat.completions.create(
    messages=[
      {
        "role": "user",
        "content": prompt,
      }
    ],
    model="gpt-4o",
  )

  answer = chat_completion.choices[0].message.content
  return answer


# %%


def batch_compute_relevance_score(abstracts: List[str]) -> List[RelevanceScore]:
  """Compute relevance scores for a batch of abstracts."""
  prompt = batch_relevance_score_template(abstracts)
  reply = call_gpt_4o(prompt)
  relevance_scores = re_batch_relevance_extraction.findall(reply)

  assert len(relevance_scores) == len(abstracts) - 1
  return relevance_scores


def get_random_walk_successor(starting_node: Paper, all_nodes: List[Paper]) -> Paper:
  """Get a successor node by sampling from all nodes and selecting among the most relevant papers."""
  sampled_nodes = sample(all_nodes, n_adjacents)

  relevance_scores = batch_compute_relevance_score(
    [starting_node.abstract] + [adj.abstract for adj in sampled_nodes]
  )

  nodes_by_relevance = [
    adj
    for adj, score in sorted(
      zip(sampled_nodes, relevance_scores), key=lambda x: x[1], reverse=True
    )
  ]
  top_nodes = nodes_by_relevance[:top_n]

  successor_node = sample(top_nodes)
  return successor_node


def generate_contribution_chain(nodes: List[Paper], length: int) -> List[Paper]:
  """Generate chain of papers by following a pseudo-random walk."""
  n0 = sample(nodes)
  chain = [n0]
  for _ in range(length):
    n = get_random_walk_successor(chain[-1], mock_arxiv_input)
    nodes.append(n)
  return chain


n_adjacents = 10
top_n = 5

mock_arxiv_input: List[Paper] = [
  Paper(abstract=f"abstract {i}", title=f"title {i}") for i in range(100)
]

generate_contribution_chain(mock_arxiv_input, length=5)
