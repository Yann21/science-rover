import arxiv
import json
from datetime import datetime


# Define the categories
categories = [
    "cs.AI",  # Computer Science - Artificial Intelligence
    "cs.LG",  # Computer Science - Machine Learning
    "math.GM",  # Mathematics - General Mathematics
    "physics.gen-ph",  # Physics - General Physics
    "econ.EM",  # Economics - Econometrics
    "q-bio.GN",  # Quantitative Biology - Genomics
    "stat.ME",  # Statistics - Methodology
    "astro-ph.GA",  # Astrophysics - Galaxy Astrophysics
    "cond-mat.mtrl-sci",  # Condensed Matter - Materials Science
    "nlin.AO"  # Nonlinear Sciences - Adaptation and Self-Organizing
]


if __name__ == "__main__":
    all_papers = []
    today = datetime.now().strftime("%Y%m%d%H%M")

    client = arxiv.Client()

    for category in categories:
        search = arxiv.Search(
            query=f"cat:{category} AND submittedDate:[201801010600 TO {today}]",
            max_results=200,
            sort_by=arxiv.SortCriterion.Relevance
        )

        print(f"Category {category}...")
        i = 0
        for i, result in enumerate(client.results(search)):
            paper_data = {
                "title": result.title,
                "abstract": result.summary
            }
            all_papers.append(paper_data)

        print(f"Retrieved {i+1} papers")

    # Write the papers data to a JSON file
    with open("arxiv_papers.json", "w", encoding='utf-8') as f:
        json.dump(all_papers, f, ensure_ascii=False)

    print(f"Successfully saved {len(all_papers)} papers' data to arxiv_papers.json")