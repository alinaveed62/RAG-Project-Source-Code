"""Query preprocessing: abbreviation expansion and normalization."""

import re

# Map of KCL and Informatics abbreviations to their full forms.
KCL_ABBREVIATIONS: dict[str, str] = {
    "EC": "Extenuating Circumstances",
    "ECs": "Extenuating Circumstances",
    "PGT": "Postgraduate Taught",
    "PGR": "Postgraduate Research",
    "UG": "Undergraduate",
    "SSP": "Student Support Plan",
    "KEATS": "King's E-Learning and Teaching Service",
    "KCL": "King's College London",
    "FYP": "Final Year Project",
    "MSc": "Master of Science",
    "BSc": "Bachelor of Science",
    "BOS": "Board of Studies",
    "SSLC": "Staff-Student Liaison Committee",
    "T&A": "Teaching and Assessment",
    "PA": "Personal Academic",
    "DoE": "Director of Education",
    "SEN": "Special Educational Needs",
    "SITS": "Student Information and Teaching System",
}


class QueryProcessor:
    """Preprocesses user queries for better retrieval."""

    def __init__(
        self,
        abbreviations: dict[str, str] | None = None,
        enable_expansion: bool = True,
    ):
        self.abbreviations = abbreviations or KCL_ABBREVIATIONS
        self.enable_expansion = enable_expansion

    def process(self, query: str) -> str:
        """Expand abbreviations and normalize the query.

        Args:
            query: Raw user query string.

        Returns:
            Processed query with abbreviations expanded (if enable_expansion)
            and whitespace normalized.
        """
        query = query.strip()
        if self.enable_expansion:
            query = self._expand_abbreviations(query)
        query = self._normalize_whitespace(query)
        return query

    def _expand_abbreviations(self, text: str) -> str:
        """Append expanded forms of detected abbreviations.

        Deduplicates by expansion value so a query that carries both
        EC and ECs does not emit "Extenuating Circumstances"
        twice in the parenthetical tail.
        """
        seen: set[str] = set()
        expansions: list[str] = []
        for abbr, full in self.abbreviations.items():
            # Match whole-word abbreviations; case-sensitive so that
            # common words like "EC" are not expanded inside "check".
            pattern = rf"\b{re.escape(abbr)}\b"
            if re.search(pattern, text) and full not in seen:
                seen.add(full)
                expansions.append(full)

        if expansions:
            text = text + " (" + ", ".join(expansions) + ")"
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces and strip."""
        return re.sub(r"\s+", " ", text).strip()
