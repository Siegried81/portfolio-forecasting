"""FastAPI JSON API package.

Thin HTTP layer over the existing src/ functions that the Streamlit app
(app.py) already uses. Additive only: the API imports and calls those
functions unchanged, so the two front-ends can never drift apart.
"""
