"""
app.py — Flask Routes
Thin HTTP layer that delegates to recommender.py and utils.py.
No ML or data-processing logic lives here.
"""

import logging
import traceback

from flask import Flask, jsonify, render_template, request

from recommender import MovieNotFoundError, Recommender
from utils import PrettyPrinter, TMDBClient

app = Flask(__name__)

# Module-level instantiation so they're ready for every request
recommender = Recommender()
tmdb_client = TMDBClient()
pretty_printer = PrettyPrinter()


@app.route("/")
def index():
    """Health check and main page.

    Returns JSON {"status": "ok"} when the client requests application/json,
    otherwise renders templates/index.html for browser clients.
    """
    if request.accept_mimetypes.best_match(["application/json", "text/html"]) == "application/json":
        return jsonify({"status": "ok"})
    return render_template("index.html")


@app.route("/recommend")
def recommend():
    """Return JSON array of up to 10 enriched recommended movies."""
    title = request.args.get("movie")
    if not title:
        return jsonify({"error": "missing 'movie' parameter"}), 400

    try:
        recs = recommender.recommend(title)
        titles = [r["title"] for r in recs]
        enriched = tmdb_client.enrich(titles)
        # Merge similarity_score into enriched results
        score_map = {r["title"]: r["similarity_score"] for r in recs}
        for item in enriched:
            item["similarity_score"] = score_map.get(item["title"], 0.0)
        return app.response_class(
            response=pretty_printer.serialize(enriched),
            status=200,
            mimetype="application/json",
        )
    except MovieNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception:
        logging.error("Unexpected error in /recommend:\n%s", traceback.format_exc())
        return jsonify({"error": "internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True)
