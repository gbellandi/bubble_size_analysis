import unittest

from bubblekicker.bubblekicker import Logger

class TestLogger(unittest.TestCase):

    def test_addition(self):
        """test the addition of a single message"""
        self.testlog = Logger()
        self.testlog.add_log("new action")
        self.assertEqual(self.testlog.log, ["new action"])

    def test_last_log(self):
        """test printing the last message"""
        self.testlog = Logger()
        self.testlog.add_log("new action 1")
        self.testlog.add_log("new action 2")
        self.assertEqual(self.testlog.get_last_log(), "new action 2")

    def test_clear(self):
        """test the clearing of the logs"""
        self.testlog = Logger()
        self.testlog.add_log("new action 1")
        self.testlog.add_log("new action 2")
        self.testlog.clear_log()
        self.assertEqual(self.testlog.log, [])


