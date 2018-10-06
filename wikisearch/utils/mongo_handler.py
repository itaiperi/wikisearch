from pymongo import MongoClient


class MongoHandler:
    def __init__(self, database_name, collection_name):
        self._mongo_client = MongoClient()
        self._collection = self._mongo_client.get_database(database_name).get_collection(collection_name)

    def get_all_pages(self):
        return self._collection.find()

    def project_page_by_field(self, field_name):
        return self._collection.find({}, {field_name: True})

    """ 
        Creates a new database and a new collection in it.
        Returns the created collection
        --- Deletes the database if already exist!!! ----
    """
    def create_database_collection_with_data(self, database_name, collection_name, data):
        db_names = self._mongo_client.list_database_names()
        if database_name in db_names:
            self._mongo_client.drop_database(database_name)
        collection = self._mongo_client[database_name][collection_name]
        collection.insert_many(data)
