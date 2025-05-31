from google.cloud import storage

def create_bucket(bucket_name):
    """Creates a new bucket."""

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    bucket.storage_class = "COLDLINE"
    
    new_bucket = storage_client.create_bucket(bucket, location="us")

    print(
        "Created bucket {} in {} with storage class {}".format(
            new_bucket.name, new_bucket.location, new_bucket.storage_class
        )
    )
    return new_bucket

if __name__ == "__main__":
    bucket_name = "example-bucket"
    create_bucket(bucket_name)
