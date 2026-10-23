"""Vercel serverless entry point; reuse the local app's request handler."""

from app.cad_risk_app import CADRiskHandler


class handler(CADRiskHandler):
    pass
