from pymongo import MongoClient


class MongoHandler:
    def __init__(self, database_name, collection_name):
        self._mongo_client = MongoClient()
        # TODO: get the collection through the function 'get_collection' instead
        self._collection = self._mongo_client.get_database(database_name).get_collection(collection_name)

    def get_all_pages(self):
        return self._collection.find()

    def project_page_by_field(self, field_name):
        return self._collection.find({}, {field_name: True})

    def get_collection(self, database, collection):
        return self._mongo_client[database][collection]

    """
        Updates a document in the given database's collection. 
        If the document doesn't exist, insert it (the parameter 'upsert' in charge of that)        
    """
    def update_a_document(self, database, collection, title, document):
        filter_title = {'title': title}
        updated_value = {"$set": document}
        self._mongo_client[database][collection].update_one(filter_title, updated_value, upsert=True)

    """
        Creates a new database and a new collection in it.
        Returns the created collection
        --- Deletes the database if already exist!!! ----
    """
    # TODO: Split this function to several functions
    def create_database_collection_with_data(self, database_name, collection_name, data):
        db_names = self._mongo_client.list_database_names()
        if database_name in db_names:
            self._mongo_client.drop_database(database_name)
        collection = self._mongo_client[database_name][collection_name]
        collection.insert_many(data)

    """
        Creates a new database. If was already exist, deletes it and recreate
    """
    def create_database(self, database_name):
        db_names = self._mongo_client.list_database_names()
        if database_name in db_names:
            self._mongo_client.drop_database(database_name)
        collection = self._mongo_client[database_name]
