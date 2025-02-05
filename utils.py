from typing import List
from dataclasses import dataclass
import os
from openai import OpenAI
import logging

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI()


@dataclass
class Paper:
  abstract: str
  title: str
  category: str


def pprint_chain(chain: List[Paper]):
  for i, paper in enumerate(chain):
    print(f"Paper {i}: {paper.title}\n{paper.abstract}\n")


def call_gpt_4o(prompt):
  chat_completion = client.chat.completions.create(
    messages=[
      {
        "role": "user",
        "content": prompt,
      }
    ],
    model="gpt-4o-mini",
  )

  answer = chat_completion.choices[0].message.content
  return answer


def get_logger():
  # Ensure logs directory exists
  os.makedirs("logs", exist_ok=True)

  # Create a new logger instance
  logger = logging.getLogger("my_logger")  # Use a unique name to avoid conflicts
  logger.setLevel(logging.DEBUG)  # Set logger level

  # Remove all handlers if they exist (prevents duplicate logs)
  while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])

  # Create a file handler (overwrite each run)
  file_handler = logging.FileHandler("logs/app.log", mode="w")
  file_handler.setLevel(logging.DEBUG)

  # Create a stream handler (console output)
  console_handler = logging.StreamHandler()
  console_handler.setLevel(logging.DEBUG)

  # Define the log format
  formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
  file_handler.setFormatter(formatter)
  console_handler.setFormatter(formatter)

  # Add handlers to logger
  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  # 🔥 Suppress noisy third-party loggers (requests, OpenAI, httpx, urllib3)
  for noisy_logger in ["httpx", "urllib3", "requests", "openai"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

  logger.info("Logger initialized successfully!")

  return logger
