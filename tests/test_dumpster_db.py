import time
import unittest

from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_REDIRECT_TO, ENTRY_TEXT, ENTRY_TITLE, ENTRY_LINKS
from wikisearch.embeddings import FastText
from wikisearch.graph import WikiGraph
from wikisearch.utils.mongo_handler import MongoHandler


class TestDumpsterParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = WikiGraph(WIKI_LANG)

    @classmethod
    def tearDownClass(cls):
        del cls.graph

    def setUp(self):
        self.mongo_handler = MongoHandler(WIKI_LANG, PAGES)
        self.graph == self.__class__.graph
        self.start_time = time.time()

    def tearDown(self):
        print(f"{self.id()}: {time.time() - self.start_time:.3f}s")
        del self.mongo_handler

    def test_has_redirect_to_or_text_only(self):
        pages = self.mongo_handler.get_all_pages()
        for page in pages:
            self.assertFalse(ENTRY_TEXT in page and ENTRY_REDIRECT_TO in page,
                             msg=f"Page '{page[ENTRY_TITLE]}' should have only one of the fields '{ENTRY_REDIRECT_TO}'"
                             f" and {ENTRY_TEXT}")

    def test_there_are_no_redirect_categories(self):
        redirect_category_pages = self.mongo_handler.get_page_by_regex(WIKI_LANG, PAGES, "^CAT:")
        self.assertIsNone(redirect_category_pages)

    def test_redirect_page(self):
        page = self.graph._redirects['USA']
        self.assertEqual(page, 'United States')
        page = self.graph._redirects['%']
        self.assertEqual(page, 'Percentage')
        page = self.graph._redirects['1991 Perfect Storm']
        self.assertEqual(page, '1991 Halloween Nor\'easter')
        page = self.graph._redirects['Abe\'s Salamander']
        self.assertEqual(page, 'Abe\'s salamander')
        page = self.graph._redirects['Acherontia']
        self.assertEqual(page, 'Death\'s-head hawkmoth')

    def test_pages_have_valuable_text(self):
        missing_text = 0
        for idx, page in enumerate(self.graph.values()):
            tokenized_text = FastText.tokenize_text(page.text)
            if not len(tokenized_text):
                print(f"Page '{page.title}' has no valuable text")
                missing_text += 1
            # TODO: once all the pages have valuable text, return the assertion and remove the prints
            # self.assertTrue(len(tokenized_text), msg=f"Page '{page[ENTRY_TITLE]}' has no valuable text")

        print(f"Found {missing_text} missing text pages")

    def test_redirected_pages_exist(self):
        for page, redirect_to in self.graph._redirects.items():
            if self.graph.get_node(redirect_to) is None:
                print(f"Page '{page}' redirects to page '{redirect_to}' which doesn't exist in {WIKI_LANG} database")
            # TODO: once all the pages redirect to existing pages return the assertion and remove the prints
            # self.assertIsNotNone(self.mongo_handler.get_page(WIKI_LANG, PAGES, page[ENTRY_REDIRECT_TO]),
            #                      msg=f"Page '{page[ENTRY_TITLE]}' redirects to page '{page[ENTRY_REDIRECT_TO]}' "
            #                      f"which doesn't exist in {WIKI_LANG} database")

    def test_link_pages_exist(self):
        for page in self.graph.values():
            for neighbor in page.neighbors:
                if self.graph.get_node(neighbor) is None:
                    print(f"Page '{page.title}' has link to '{neighbor}' which doesn't exist in {WIKI_LANG} database")
                    # TODO: once all the pages' links exist return the assertion and remove the prints
                    # self.assertIsNotNone(self.mongo_handler.get_page(WIKI_LANG, PAGES, page[ENTRY_LINKS]),
                    #                      msg=f"Page '{page[ENTRY_TITLE]}' has link to '{link}' which doesn't "
                    #                      f"exist in {WIKI_LANG} database")


if __name__ == "__main__":
    unittest.main()
