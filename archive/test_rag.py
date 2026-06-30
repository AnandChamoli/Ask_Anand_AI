from rag_engine import retrieve_context

questions = [
    "Who is Anand Chamoli?",
    "Does Anand have GCP experience?",
    "Tell me about CARE Direct 24x7.",
    "Why should we hire Anand?",
    "What analytics projects has Anand completed?"
]

for question in questions:
    print("\n" + "=" * 80)
    print("QUESTION:")
    print(question)

    results = retrieve_context(question)

    print("\nTOP MATCHES:\n")

    for i, doc in enumerate(results["documents"][0]):
        print(f"Match {i+1}:")
        print(doc[:500])
        print("\n" + "-" * 80)
