import os


def db_params():
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "vector_db_1"),
        "user": os.getenv("DB_USER", "charlie"),
        "password": os.getenv("DB_PASSWORD", "malicay"),
    }
