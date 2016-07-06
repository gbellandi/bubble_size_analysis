import unittest

from bubblekicker import BubbleKicker


class TestBubbleKicker(unittest.TestCase):

    def setUp(self):
        self.dummy_image = None # Provide a very simple image here

    def test_dilate(self):
        """test the dilate handling"""
        # add function and check if outcome is ok by testing against known
        BubbleKicker()
        # outcome:

        self.assertEquals()