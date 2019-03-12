import re

import pymongo
from pymongo import MongoClient, UpdateOne

from wikisearch.consts.mongo import ENTRY_TITLE


class MongoHandler:
    """
        A utility class to communicate with mongo database.
        Each Mongo Handler is per database and collection. If the database or the collection doesn't exist the
        handler will create them locally.
    """

    def __init__(self, database, collection):
        self._mongo_client = MongoClient()
        self._collection = self._mongo_client[database][collection]

    def get_all_documents(self, projection=None):
        return self.get_pages({}, projection=projection)

    def get_pages(self, filter, projection=None):
        return self._collection.find(filter, projection=projection)

    def get_page(self, title, projection=None):
        return self._collection.find_one({'title': title}, projection=projection)

    def get_page_by_regex(self, title_regex, projection=None):
        regex = re.compile(title_regex)
        return self.get_page({'title': regex}, projection=projection)

    def update_page(self, page):
        """
        Updates a document by it the page title. If the document doesn't exist, insert it
        (the parameter 'upsert' is in charge of that)
        :param page: the updated page
        """
        filter_title = {'title': page[ENTRY_TITLE]}
        updated_value = {"$set": page}
        self._collection.update_one(filter_title, updated_value, upsert=True)

    def is_empty_collection(self):
        return self._collection.count_documents({}) == 0

    def delete_collection_data(self):
        self.delete_pages({})

    def delete_pages(self, filter):
        self._collection.delete_many(filter)

    def insert_data(self, data):
        self._collection.insert_many(data)

    @staticmethod
    def update_page_request(page):
        filter_title = {'title': page[ENTRY_TITLE]}
        updated_value = {"$set": page}
        return UpdateOne(filter_title, updated_value, upsert=True)

    def bulk_write(self, requests):
        self._collection.bulk_write(requests)

    def create_title_index(self):
        self._collection.create_index([("title", pymongo.ASCENDING)], unique=True)
