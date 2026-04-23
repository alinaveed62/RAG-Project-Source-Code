"""Sanitized HTML samples mimicking real KEATS Moodle pages for integration testing."""

# Moodle book chapter page - content inside #region-main with .block elements
BOOK_CHAPTER_HTML = """
<!DOCTYPE html>
<html>
<head><title>Teaching &amp; Assessment - KEATS</title></head>
<body>
<nav class="navbar">Navigation bar</nav>
<div id="page">
    <div id="region-main">
        <div class="block block_book_toc">
            <h3>Table of contents</h3>
            <ul>
                <li><a href="/mod/book/view.php?id=123&chapterid=1">Introduction</a></li>
                <li><a href="/mod/book/view.php?id=123&chapterid=2">Assessment</a></li>
            </ul>
        </div>
        <div class="book_content">
            <h1>Teaching &amp; Assessment</h1>
            <h2>Assessment Methods</h2>
            <p>The Department of Informatics uses a variety of assessment methods
            including coursework, examinations, and practical assignments. Each module
            specifies its own assessment criteria and weighting in the module descriptor.</p>
            <h3>Coursework Submission</h3>
            <p>All coursework must be submitted electronically via KEATS by the published
            deadline. Late submissions will incur penalties as described in the College
            regulations. Extensions may be granted through the extenuating circumstances
            procedure.</p>
            <h3>Examination Regulations</h3>
            <p>Examinations are conducted in accordance with King's College London
            regulations. Students must present their student ID at all examinations.
            Any form of academic misconduct will be dealt with under the College's
            Academic Misconduct Policy.</p>
            <table>
                <tr><th>Assessment Type</th><th>Weight</th></tr>
                <tr><td>Coursework</td><td>40%</td></tr>
                <tr><td>Final Exam</td><td>60%</td></tr>
            </table>
        </div>
        <div class="activity-navigation">
            <a href="prev">Previous</a>
            <a href="next">Next</a>
        </div>
    </div>
</div>
<footer>Footer content</footer>
</body>
</html>
"""

# Moodle page with content inside .block elements (the problematic case)
PAGE_WITH_BLOCK_CONTENT_HTML = """
<!DOCTYPE html>
<html>
<head><title>Student Support - KEATS</title></head>
<body>
<nav class="navbar">Nav</nav>
<div id="page">
    <div id="region-main">
        <div class="block">
            <div class="content">
                <h1>Student Support &amp; Wellbeing</h1>
                <p>King's College London provides comprehensive support services for all
                students. The Department of Informatics has dedicated support channels
                including personal tutors, disability liaison officers, and mental health
                first aiders.</p>
                <h2>Personal Tutors</h2>
                <p>Every student is assigned a personal tutor at the start of their
                programme. Your personal tutor is your first point of contact for
                academic and pastoral support. You should meet with your personal tutor
                at least once per term.</p>
                <h2>Disability Support</h2>
                <p>Students with disabilities or specific learning differences should
                register with the Disability Advisory Service. The department has a
                dedicated Disability Liaison Officer who can help arrange reasonable
                adjustments.</p>
            </div>
        </div>
    </div>
</div>
<footer>Footer</footer>
</body>
</html>
"""

# Simple Moodle page with normal content
SIMPLE_PAGE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Attendance Policy - KEATS</title></head>
<body>
<nav class="navbar">Navigation</nav>
<div id="page">
    <div id="region-main">
        <h1>Attendance Policy</h1>
        <div class="content">
            <p>Students are expected to attend all scheduled teaching activities
            including lectures, tutorials, and laboratory sessions. The College
            monitors engagement through various mechanisms.</p>
            <h2>Engagement Monitoring</h2>
            <p>The department uses engagement monitoring points throughout the
            academic year. Students who miss multiple monitoring points may be
            contacted by the Student Support team.</p>
            <ul>
                <li>Lecture attendance is recorded via QR code sign-in</li>
                <li>Tutorial attendance is recorded by the tutor</li>
                <li>Laboratory attendance is mandatory</li>
            </ul>
        </div>
    </div>
</div>
<footer>Footer content</footer>
</body>
</html>
"""

# Page that produces empty content after cleaning
EMPTY_AFTER_CLEANING_HTML = """
<!DOCTYPE html>
<html>
<head><title>Empty Page - KEATS</title></head>
<body>
<nav class="navbar">Navigation</nav>
<div id="page">
    <div id="region-main">
        <script>console.log('test');</script>
        <style>.hidden { display: none; }</style>
        <div class="activity-navigation">
            <a href="prev">Previous</a>
            <a href="next">Next</a>
        </div>
        <span class="sr-only">Screen reader only</span>
    </div>
</div>
<footer>Footer</footer>
</body>
</html>
"""
