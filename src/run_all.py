import json
from pathlib import Path

from data_categorize import categorize_articles
from data_cluster import cluster_articles, save_clusters_to_json
from data_compare import analyse_clustered_file, summarize_relations
from data_parse import parse_all_documents
from data_filter import filter_articles


if __name__ == "__main__":
    sources = ["EBA", "FIVA_MOK"]
    categories = [
        "credit_risk",
        "liquidity_risk",
        "market_risk",
        "operational_risk",
        "compliance_risk",
    ]

    for source in sources:
        #########
        # PARSE #
        #########

        results = parse_all_documents(source)

        print("Saving parse results")
        save_dir = (
            Path(__file__).parent.parent.resolve() / "data" / "intermediate" / "parsed"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        filename = f"all_{source}_parsed.json"
        with open(save_dir / filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        print(f"Saved {len(results)} documents to {filename}")

        ##########
        # FILTER #
        ##########

        relevant_articles, unrelated_articles = filter_articles(source, verbose=True)

        # Save to file
        print("Saving filter results")
        save_dir = (
            Path(__file__).parent.parent.resolve()
            / "data"
            / "intermediate"
            / "filtered"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        relevant_filename = f"credit_related_{source}.json"
        unrelated_filename = f"unrelated_{source}.json"

        with open(save_dir / relevant_filename, "w", encoding="utf-8") as f:
            json.dump(relevant_articles, f, ensure_ascii=False, indent=4)

        with open(save_dir / unrelated_filename, "w", encoding="utf-8") as f:
            json.dump(unrelated_articles, f, ensure_ascii=False, indent=4)

        print(
            f"Saved {len(relevant_articles)} relevant articles to {relevant_filename}"
        )
        print(
            f"Saved {len(unrelated_articles)} unrelated articles to {unrelated_filename}"
        )

        ##############
        # CATEGORIZE #
        ##############

        categorized_articles, unclassified_count = categorize_articles(
            source, verbose=True
        )

        # Save to files
        print("Saving categorization results")
        save_dir = (
            Path(__file__).parent.parent.resolve()
            / "data"
            / "intermediate"
            / "categorized"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each category to separate files
        for category, articles in categorized_articles.items():
            category_filename = f"{category.lower()}_{source}.json"
            with open(save_dir / category_filename, "w", encoding="utf-8") as f:
                json.dump(articles, f, ensure_ascii=False, indent=4)
            print(f"Saved {len(articles)} {category} articles to {category_filename}")

    for category in categories:
        ###########
        # CLUSTER #
        ###########

        clusters = cluster_articles(
            sources,
            category,
            tfidf_threshold=0.2,
            llm_threshold=0.84,
            early_exit=0.91,
            verbose=True,
        )

        # Save to file
        print("Saving results")
        save_dir = (
            Path(__file__).parent.parent.resolve()
            / "data"
            / "intermediate"
            / "clustered"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        if clusters:
            save_clusters_to_json(clusters, save_dir / f"{category}.json")
            print(f"Successfully clustered articles into {len(clusters)} groups")
        else:
            print(f"No clusters created for category: {category}")

        ###########
        # COMPARE #
        ###########

        results = analyse_clustered_file(category)
        summary = summarize_relations(results)

        # save all article-level relations
        with open(
            Path(__file__).parent.parent.resolve() / "results" / f"{category}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(len(results), "article pairs analysed (including bloat)")
        print("Overlaps (used in metrics):", summary["overlap"])
        print("Contradictions (used in metrics):", summary["contradiction"])
        print("Bloat (ignored in metrics):", summary["bloat"])
        print(
            "Total used for metrics (overlap + contradiction):", summary["total_metric"]
        )
        print("Total pairs including bloat:", summary["total_all"])
