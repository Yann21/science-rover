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
from retriever import categories
from prompts import extract_auto_criticism_score, auto_criticism_prompt

load_dotenv()
logger = get_logger()


def batch_compute_relevance_score(original_paper, papers: List[str]) -> List[int]:
  """Compute relevance scores for a batch of abstracts."""
  messages = batch_relevance_score_prompt(original_paper, papers)
  reply = call_gpt_4o(messages)
  relevance_scores = re_batch_relevance_extraction.findall(reply)

  assert len(relevance_scores) == len(papers), (
    f"Expected {len(papers)} relevance scores, got {len(relevance_scores)}:\nReply: {reply}"
  )
  return relevance_scores


def get_random_walk_successor(starting_node: Paper, all_nodes: List[Paper]) -> Paper:
  """Get a successor node by sampling from all nodes and selecting among the most relevant papers."""
  sampled_nodes = sample(all_nodes, k=n_adjacents)

  relevance_scores = batch_compute_relevance_score(starting_node, sampled_nodes)
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
  ml_nodes: List[Paper] = [node for node in nodes if node.category == "cs.LG"]
  n0: List[Paper] = sample(ml_nodes, k=1)
  chain = n0

  # Heuristic of staying in the same domain
  n1: Paper = get_random_walk_successor(chain[-1], nodes)
  domain = n1.category
  nodes = [node for node in nodes if node.category in [domain, "cs.LG"]]

  for _ in range(length - 2):
    n: Paper = get_random_walk_successor(chain[-1], nodes)
    chain.append(n)
  return chain


def generate_contribution(chain: List[Paper]) -> str:
  """Generate a contribution from a chain of papers."""
  messages = contribution_prompt(chain)
  contribution = call_gpt_4o(messages)

  with open("logs/contribution.log", "a", encoding="utf-8") as f:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score = get_auto_criticism_score(contribution)
    f.write(f"\n\n=== {ts} ===" + f"\n{score}\n\n\n" + contribution)

  return contribution


def get_auto_criticism_score(contribution: str) -> int:
  messages = auto_criticism_prompt(contribution)
  reply = call_gpt_4o(messages)
  score = extract_auto_criticism_score(reply)
  return score


# %%
n_adjacents = 20
top_n = 1
chain_length = 2

with open("arxiv_papers.json", "r", encoding="utf-8") as f:
  arxiv_papers = json.load(f)
  arxiv_papers = [Paper(**paper) for paper in arxiv_papers]


# %%
for _ in range(5):
  chain = generate_contribution_chain(arxiv_papers, length=chain_length)
  # pprint_chain(chain)
  contrib = generate_contribution(chain)
  # print(contrib)
