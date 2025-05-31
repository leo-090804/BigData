from google.cloud import storage

def list_buckets():
    """Lists all buckets."""

    storage_client = storage.Client()
    print("Authorized User")
    buckets = storage_client.list_buckets()

    for bucket in buckets:
        print(bucket.name)

if __name__ == "__main__":
    list_buckets()
