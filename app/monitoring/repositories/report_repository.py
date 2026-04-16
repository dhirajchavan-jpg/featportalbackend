# app/monitoring/repositories/report_repository.py
from app.database import db
from bson import ObjectId

class ReportRepository:
    collection = db["rag_reports"]

    @staticmethod
    async def save(report: dict) -> dict:
        result = await ReportRepository.collection.insert_one(report)

        #  Convert ObjectId → string
        report["_id"] = str(result.inserted_id)
        return report

    @staticmethod
    async def fetch(report_type: str, limit=5) -> list:
        cursor = ReportRepository.collection.find(
            {"report_type": report_type}
        ).sort("generated_at", -1).limit(limit)

        reports = await cursor.to_list(length=limit)

        #  Convert ObjectId → string for all documents
        for r in reports:
            r["_id"] = str(r["_id"])

        return reports
