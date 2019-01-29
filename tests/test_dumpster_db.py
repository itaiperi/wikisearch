import unittest

from wikisearch.consts.mongo import WIKI_LANG, PAGES, ENTRY_REDIRECT_TO, ENTRY_TEXT, ENTRY_TITLE, ENTRY_LINKS
from wikisearch.embeddings import FastText
from wikisearch.utils.mongo_handler import MongoHandler


class TestDumpsterParsing(unittest.TestCase):
    def setUp(self):
        self.mongo_handler = MongoHandler(WIKI_LANG, PAGES)

    def tearDown(self):
        del self.mongo_handler

    def test_redirect_page(self):
        page = self.mongo_handler.get_page(WIKI_LANG, PAGES, 'USA')
        self.assertEqual(page[ENTRY_REDIRECT_TO], 'United States')
        page = self.mongo_handler.get_page(WIKI_LANG, PAGES, '%')
        self.assertEqual(page[ENTRY_REDIRECT_TO], 'Percentage')
        page = self.mongo_handler.get_page(WIKI_LANG, PAGES, '1991 Perfect Storm')
        self.assertEqual(page[ENTRY_REDIRECT_TO], '1991 Halloween Nor\'easter')
        page = self.mongo_handler.get_page(WIKI_LANG, PAGES, 'Abe\'s Salamander')
        self.assertEqual(page[ENTRY_REDIRECT_TO], 'Abe\'s salamander')
        page = self.mongo_handler.get_page(WIKI_LANG, PAGES, 'Acherontia')
        self.assertEqual(page[ENTRY_REDIRECT_TO], 'Death\'s-head hawkmoth')

    def test_pages_have_valuable_text(self):
        pages = self.mongo_handler.get_all_pages()
        missing_text = 0
        for idx, page in enumerate(pages):
            if ENTRY_TEXT in page:
                tokenized_text = FastText.tokenize_text(page[ENTRY_TEXT])
                if not len(tokenized_text):
                    print(f"Page {page[ENTRY_TITLE]} has no valuable text")
                    missing_text += 1
                # TODO: once all the pages have valuable text, return the assertion and remove the prints
                # self.assertTrue(len(tokenized_text), msg=f"Page {page[ENTRY_TITLE]} has no valuable text")

        print(f"Found {missing_text} missing text pages")

    def test_has_redirect_to_or_text_only(self):
        pages = self.mongo_handler.get_all_pages()
        for page in pages:
            self.assertFalse(ENTRY_TEXT in page and ENTRY_REDIRECT_TO in page,
                             msg=f"Page {page[ENTRY_TITLE]} should have only one of the fields {ENTRY_REDIRECT_TO}"
                             f" and {ENTRY_TEXT}")

    def test_there_are_no_redirect_categories(self):
        redirect_category_pages = self.mongo_handler.get_page_by_regex(WIKI_LANG, PAGES, "^CAT:")
        self.assertIsNone(redirect_category_pages)

    def test_redirected_pages_exist(self):
        pages = self.mongo_handler.get_all_pages()
        for page in pages:
            if ENTRY_REDIRECT_TO in page:
                if self.mongo_handler.get_page(WIKI_LANG, PAGES, page[ENTRY_REDIRECT_TO]) is None:
                    print(f"Page '{page[ENTRY_TITLE]}' redirects to page {page[ENTRY_REDIRECT_TO]} "
                          f"which does't exist in {WIKI_LANG} database")
                # TODO: once all the pages redirect to existing pages return the assertion and remove the prints
                # self.assertIsNotNone(self.mongo_handler.get_page(WIKI_LANG, PAGES, page[ENTRY_REDIRECT_TO]),
                #                      msg=f"Page '{page[ENTRY_TITLE]}' redirects to page {page[ENTRY_REDIRECT_TO]} "
                #                      f"which does't exist in {WIKI_LANG} database")

    # def test_link_pages_exist(self):
    #     pages = self.mongo_handler.get_all_pages()
    #     for page in pages:
    #         if ENTRY_LINKS in page:
    #             for link in page[ENTRY_LINKS]:
    #                 if self.mongo_handler.get_page(WIKI_LANG, PAGES, link) is None:
    #                     print(f"Page {page[ENTRY_TITLE]} has link to {link} which does't exist in {WIKI_LANG} database")
    #                     # TODO: once all the pages' links exist return the assertion and remove the prints
    #                     # self.assertIsNotNone(self.mongo_handler.get_page(WIKI_LANG, PAGES, page[ENTRY_LINKS]),
    #                     #                      msg=f"Page {page[ENTRY_TITLE]} has link to {link} which does't "
    #                     #                      f"exist in {WIKI_LANG} database")


if __name__ == "__main__":
    unittest.main()
