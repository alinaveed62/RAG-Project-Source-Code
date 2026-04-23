"""Flask routes for the RAG chatbot."""

from flask import Blueprint, current_app, render_template, request

from flask_app.feedback.logger import log_feedback
from rag_pipeline.generation.citation_parser import strip_citations

main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET", "POST"])
def index():
    """Handle question submission and display answers."""
    if request.method == "GET":
        return render_template("index.html")

    question = request.form.get("question", "").strip()
    if not question:
        return render_template("index.html", error="Please enter a question.")

    pipeline = current_app.config.get("PIPELINE")
    if pipeline is None:
        return render_template(
            "index.html",
            error="The RAG pipeline is not loaded. Please start the server with a pipeline.",
        )

    response = pipeline.answer(question)

    # Remove the raw [Source: cN] markers from the prose shown to the user,
    # and pass the validated citation list separately so the template can
    # render chips linked to the sources panel.
    clean_answer = strip_citations(response.answer)
    citations = [c.model_dump() for c in response.citations]

    return render_template(
        "answer.html",
        question=response.question,
        answer=clean_answer,
        sources=[s.model_dump() for s in response.sources],
        citations=citations,
        retrieval_time_ms=response.retrieval_time_ms,
        rerank_time_ms=response.rerank_time_ms,
        generation_time_ms=response.generation_time_ms,
    )


def _parse_rating(raw: str | None) -> int:
    """Coerce a form field into a 1 to 5 star rating, or 0 if invalid.

    Rejects out-of-range values (for example 99) and non-integer strings
    so the feedback store only receives clean values.
    """
    if raw is None:
        return 0
    try:
        value = int(raw)
    except (ValueError, TypeError):
        return 0
    if not 1 <= value <= 5:
        return 0
    return value


@main_bp.route("/feedback", methods=["POST"])
def feedback():
    """Record user feedback for a response."""
    answer_rating = _parse_rating(request.form.get("answer_rating"))
    source_rating = _parse_rating(request.form.get("source_rating"))

    data = {
        "question": request.form.get("question", ""),
        "answer_rating": answer_rating,
        "source_rating": source_rating,
        "comments": request.form.get("comments", ""),
    }

    feedback_dir = current_app.config.get("FEEDBACK_DIR")
    log_feedback(data, feedback_dir)

    return render_template("index.html", feedback_submitted=True)
