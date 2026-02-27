import requests
import re
from collections import Counter
import statistics
import time   # <-- ADDED

API_URL = "http://127.0.0.1:8000/query"
TIMEOUT = 30  


# Test questions
TEST_DATA = [

    {
        "question": "What does anisotropy mean in Transformer representations?",
        "ground_truth": "Anisotropy means token representations are concentrated in a narrow cone rather than being uniformly distributed in embedding space."
    },
    {
        "question": "Why do the authors argue anisotropy is inherent to self-attention?",
        "ground_truth": "Because even randomly initialized untrained Transformers exhibit anisotropic representations, showing it arises from the self-attention mechanism itself."
    },
    {
        "question": "What main claim is made about anisotropy in Transformers?",
        "ground_truth": "The paper argues that anisotropy is not inherently unavoidable and depends on architectural and training choices."
    },
    {
        "question": "How can architectural design affect anisotropy?",
        "ground_truth": "Changes in normalization placement and embedding design can mitigate or eliminate anisotropy."
    },
    {
        "question": "Why is learning rate warmup commonly used in Transformers?",
        "ground_truth": "Warmup is used to prevent unstable updates and optimization breakdown caused by large early Adam updates and gradient issues."
    },
    {
        "question": "How does sparse or local attention affect rank collapse?",
        "ground_truth": "Sparse attention slows down the rate of rank collapse compared to full bidirectional attention."
    }
]


def normalize(text):
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text.lower().strip())


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def compute_f1(prediction, ground_truth):
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction, ground_truth):
    return int(normalize(prediction) == normalize(ground_truth))


def compute_context_recall(context, ground_truth):
    context_tokens = set(tokenize(context))
    gt_tokens = set(tokenize(ground_truth))

    if not gt_tokens:
        return 0

    overlap = len(context_tokens & gt_tokens)
    recall = overlap / len(gt_tokens)

    return int(recall >= 0.6)


def evaluate_mode(mode):

    f1_scores = []
    em_scores = []
    context_scores = []

    print(f"\nEvaluating:\n")

    for item in TEST_DATA:

        question = item["question"]
        ground_truth = item["ground_truth"]

        try:
            response = requests.post(
                API_URL,
                params={"query": question, "mode": mode},
                timeout=TIMEOUT
            )

            response.raise_for_status()
            data = response.json()
            #added delay so that the api doesn't get overloaded
            time.sleep(1.5)   

        except Exception as e:
            print(f"❌ API Error for question: {question}")
            print("Error:", e)
            print("-" * 60)
            continue

        answer = data.get("answer", "")
        context = data.get("context", "")

        f1 = compute_f1(answer, ground_truth)
        em = compute_exact_match(answer, ground_truth)
        context_hit = compute_context_recall(context, ground_truth)

        f1_scores.append(f1)
        em_scores.append(em)
        context_scores.append(context_hit)

        print("Q:", question)
        print("Answer:", answer[:120], "...")
        print("F1:", round(f1, 3),
              "| EM:", em,
              "| Context Recall:", context_hit)
        print("-" * 60)

    if not f1_scores:
        print("No successful queries.")
        return 0, 0, 0

    avg_f1 = statistics.mean(f1_scores)
    avg_em = statistics.mean(em_scores)
    avg_context = statistics.mean(context_scores)

    print("\n===== FINAL RESULTS =====")
    print("Mode:", mode)
    print("Avg F1:", round(avg_f1, 3))
    print("Accuracy (Exact Match):", round(avg_em, 3))
    print("Context Recall@k:", round(avg_context, 3))

    return avg_f1, avg_em, avg_context


if __name__ == "__main__":

    sliding_scores = evaluate_mode("sliding")
    semantic_scores = evaluate_mode("semantic")

    print("\n========= COMPARISON =========")
    print("Sliding  -> F1:", round(sliding_scores[0], 3),
          "| Acc:", round(sliding_scores[1], 3),
          "| Recall:", round(sliding_scores[2], 3))

    print("Semantic -> F1:", round(semantic_scores[0], 3),
          "| Acc:", round(semantic_scores[1], 3),
          "| Recall:", round(semantic_scores[2], 3))