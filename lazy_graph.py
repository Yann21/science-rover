# %%
from typing import Dict, List
from random import sample
from prompts import (
  batch_relevance_score_prompt,
  re_batch_relevance_extraction,
  contribution_prompt,
)
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
from utils import Paper, call_gpt_4o, pprint_chain, get_logger
from datetime import datetime

load_dotenv()
logger = get_logger()


def batch_compute_relevance_score(abstracts: List[str]) -> List[int]:
  """Compute relevance scores for a batch of abstracts."""
  prompt = batch_relevance_score_prompt(abstracts)
  reply = call_gpt_4o(prompt)
  relevance_scores = re_batch_relevance_extraction.findall(reply)

  assert len(relevance_scores) == len(abstracts) - 1, (
    f"Expected {len(abstracts) - 1} relevance scores, got {len(relevance_scores)}:\nReply: {reply}"
  )
  return relevance_scores


def get_random_walk_successor(starting_node: Paper, all_nodes: List[Paper]) -> Paper:
  """Get a successor node by sampling from all nodes and selecting among the most relevant papers."""
  sampled_nodes = sample(all_nodes, k=n_adjacents)

  relevance_scores = batch_compute_relevance_score(
    [starting_node.abstract] + [adj.abstract for adj in sampled_nodes]
  )
  logger.info(f"Relevance scores: {relevance_scores}")

  nodes_by_relevance = [
    adj
    for adj, score in sorted(
      zip(sampled_nodes, relevance_scores), key=lambda x: x[1], reverse=True
    )
  ]
  top_nodes = nodes_by_relevance[:top_n]

  successor_node = sample(top_nodes, k=1)[0]
  return successor_node


def generate_contribution_chain(nodes: List[Paper], length: int) -> List[Paper]:
  """Generate chain of papers by following a pseudo-random walk."""
  n0: List[Paper] = sample(nodes, k=1)
  chain = n0
  for _ in range(length - 1):
    n = get_random_walk_successor(chain[-1], nodes)
    chain.append(n)
  return chain


def generate_contribution(chain: List[Paper]) -> str:
  """Generate a contribution from a chain of papers."""
  abstracts = [paper.abstract for paper in chain]
  prompt = contribution_prompt("\n\n".join(abstracts))
  contribution = call_gpt_4o(prompt)

  # log the contribution
  with open("logs/contribution.log", "a", encoding="utf-8") as f:
    # add also a timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.write(f"\n\n=== {ts} ===\n\n" + contribution)

  return contribution


# %%
n_adjacents = 20
top_n = 1
chain_length = 5

with open("arxiv_papers.json", "r", encoding="utf-8") as f:
  arxiv_papers = json.load(f)
  arxiv_papers = [Paper(**paper) for paper in arxiv_papers]


# %%
chain = generate_contribution_chain(arxiv_papers, length=chain_length)
# pprint_chain(chain)


# %%
contrib = generate_contribution(chain)
print(contrib)
