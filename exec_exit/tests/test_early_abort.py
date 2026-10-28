import unittest

from exec_exit.replay.early_abort import StreamAbortDetector, match_visible_comment


class EarlyAbortTests(unittest.TestCase):
    def detect(self, pieces, final=True):
        stream = StreamAbortDetector()
        for piece in pieces:
            match = stream.feed(piece)
            if match:
                return match
        return stream.finish() if final else stream.match

    def test_waiting_is_not_wait(self):
        self.assertIsNone(self.detect(["# wait", "ing for input\n"]))

    def test_right_boundary_must_arrive(self):
        self.assertIsNone(self.detect(["# wait"], final=False))
        self.assertEqual(self.detect(["# wait", ", continue"]).token_count, 2)

    def test_eof_boundary_no_added_token(self):
        self.assertEqual(self.detect(["# wait"]).token_count, 1)

    def test_case_indent_phrase_split(self):
        match = self.detect(["x=1\n   # LET", " ME THINK", ".\n"])
        self.assertEqual((match.token_count, match.line_number, match.phrase), (3, 2, "let me think"))

    def test_inline_comment_is_excluded(self):
        self.assertIsNone(self.detect(["x=1 # wait.\n"]))

    def test_left_boundary(self):
        self.assertIsNone(self.detect(["# await.\n# hmm_value\n"]))

    def test_all_words(self):
        for phrase in ("wait", "let's think", "let me think", "hmm", "think again", "alternatively", "but wait", "maybe"):
            with self.subTest(phrase=phrase):
                self.assertIsNotNone(self.detect(["# " + phrase + "!\n"]))

    def test_bytes_unicode_split(self):
        self.assertIsNotNone(self.detect([b"# \xe4", b"\xb8\xad wait", b" "]))

    def test_multiple_lines_in_one_token(self):
        match = self.detect(["# read\nx=1\n# maybe later\n"])
        self.assertEqual((match.token_count, match.line_number), (1, 3))

    def test_future_line_not_read(self):
        match = self.detect(["# wait ", "future token"])
        self.assertNotIn("future", match.line_prefix)

    def test_native_cumulative_text_boundary(self):
        self.assertIsNone(match_visible_comment("# wait"))
        self.assertIsNone(match_visible_comment("# waiting\n"))
        found=match_visible_comment("x=1\n# wait.")
        self.assertEqual(found["phrase"],"wait")
        self.assertEqual(found["visible_through_character"],len("x=1\n# wait."))
        self.assertEqual(match_visible_comment("# wait",final=True)["right_boundary"],"EOF")


if __name__ == "__main__":
    unittest.main()
