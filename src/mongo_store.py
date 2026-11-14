"""Optional MongoDB persistence for OptiRetail AI.

Configure Streamlit secrets:
MONGODB_URI = "mongodb+srv://..."
MONGODB_DATABASE = "optiretail"
Collections default to predictions, inventory, sales_history.
"""
from datetime import datetime, timezone

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

_CLIENT = None


def _settings(secrets):
    uri = str(secrets.get("MONGODB_URI", "")).strip()
    database = str(secrets.get("MONGODB_DATABASE", "optiretail")).strip() or "optiretail"
    return uri, database


def get_database(secrets):
    global _CLIENT
    if MongoClient is None:
        return None, "pymongo is not installed."
    uri, database = _settings(secrets)
    if not uri:
        return None, "MongoDB is not configured. Add MONGODB_URI to Streamlit Secrets."
    try:
        if _CLIENT is None:
            _CLIENT = MongoClient(uri, serverSelectionTimeoutMS=4000, connectTimeoutMS=4000)
        _CLIENT.admin.command("ping")
        return _CLIENT[database], "Connected to MongoDB."
    except Exception as exc:
        _CLIENT = None
        return None, f"MongoDB connection failed: {exc}"


def save_prediction(secrets, prediction):
    db, message = get_database(secrets)
    if db is None:
        return False, message
    record = dict(prediction)
    record["_saved_at"] = datetime.now(timezone.utc)
    try:
        db["predictions"].insert_one(record)
        return True, "Decision saved to MongoDB."
    except Exception as exc:
        return False, f"Could not save decision: {exc}"


def latest_prediction(secrets):
    db, message = get_database(secrets)
    if db is None:
        return None, message
    try:
        record = db["predictions"].find_one({}, sort=[("_saved_at", -1)])
        if record:
            record.pop("_id", None)
            record.pop("_saved_at", None)
        return record, "Loaded latest decision from MongoDB." if record else "No saved decisions found in MongoDB."
    except Exception as exc:
        return None, f"Could not load decision: {exc}"


def load_inventory(secrets):
    db, message = get_database(secrets)
    if db is None:
        return None, message
    try:
        rows = list(db["inventory"].find({}, {"_id": 0}))
        return rows, "Loaded inventory from MongoDB." if rows else "MongoDB inventory collection is empty."
    except Exception as exc:
        return None, f"Could not load inventory: {exc}"


def replace_inventory(secrets, frame):
    db, message = get_database(secrets)
    if db is None:
        return False, message
    try:
        db["inventory"].delete_many({})
        rows = frame.to_dict("records")
        if rows:
            db["inventory"].insert_many(rows)
        return True, f"Saved {len(rows)} inventory rows to MongoDB."
    except Exception as exc:
        return False, f"Could not save inventory: {exc}"


def load_sales_history(secrets, product=None):
    db, message = get_database(secrets)
    if db is None:
        return None, message
    try:
        query = {"product": product} if product else {}
        rows = list(db["sales_history"].find(query, {"_id": 0}))
        return rows, "Loaded sales history from MongoDB." if rows else "No matching sales history found in MongoDB."
    except Exception as exc:
        return None, f"Could not load sales history: {exc}"
