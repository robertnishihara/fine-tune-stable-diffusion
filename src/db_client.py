import dbm

storage_path = "storage"  # db automatically appended


class DBClient:
    def __init__(self, model: str, path: str = storage_path):
        self.path = path
        self.model = model

    def __enter__(self):
        self.db = dbm.open(self.path, "c")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def get(self, key: str) -> str:
        try:
            return self.db.get(f"{self.model}:{key}").decode()
        except AttributeError:
            raise HTTPException(
                status_code=400, detail=f"Item {self.model, key} not found"
            ) from None

    def set(self, key: str, value: str):
        self.db[f"{self.model}:{key}"] = value

    def delete(self, key: str):
        del self.db[f"{self.model}:{key}"]

    def clear_all(self):
        for key in self.db.keys():
            if key.decode().startswith(f"{self.model}:"):
                del self.db[key]

    def has_entries(self):
        for key in self.db.keys():
            if key.decode().startswith(f"{self.model}:"):
                return True
        return False

    def __getitem__(self, key: str) -> str:
        return self.get(key)

    def __setitem__(self, key: str, value: str):
        self.set(key, value)
