"""Unit-style examples for the negative review reason detector."""

import unittest

from src.models.negative_reason_detector import NegativeReasonDetector, OTHER_LABEL


class NegativeReasonDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = NegativeReasonDetector()

    def test_negated_reason_phrases_are_not_detected(self) -> None:
        result = self.detector.analyze_review(
            "The acting was not bad, and the dialogue was not awful."
        )

        self.assertEqual(result.labels, [OTHER_LABEL])
        self.assertNotIn("bad_acting", result.reasons)
        self.assertNotIn("bad_dialogue", result.reasons)

    def test_contrastive_sentence_keeps_complaint_after_but(self) -> None:
        result = self.detector.analyze_review(
            "The acting was not bad, but the plot was weak and the ending was terrible."
        )

        self.assertNotIn("bad_acting", result.labels)
        self.assertIn("weak_plot_bad_writing", result.labels)
        self.assertIn("disappointing_ending", result.labels)
        self.assertTrue(result.has_multiple_reasons)

    def test_single_weak_word_does_not_trigger_label(self) -> None:
        result = self.detector.analyze_review("It was predictable.")

        self.assertEqual(result.labels, [OTHER_LABEL])

    def test_repeated_weak_pacing_evidence_can_trigger_label(self) -> None:
        result = self.detector.analyze_review(
            "The movie was boring, slow, and repetitive."
        )

        self.assertIn("boring_slow_pacing", result.labels)
        self.assertGreaterEqual(
            result.reasons["boring_slow_pacing"]["weak_evidence_count"], 2
        )

    def test_multi_label_review_with_explicit_evidence(self) -> None:
        result = self.detector.analyze_review(
            "The acting was wooden. The dialogue was awful. The CGI looked cheap."
        )

        self.assertTrue(
            {"bad_acting", "bad_dialogue", "poor_visuals_effects"}.issubset(
                set(result.labels)
            )
        )
        self.assertTrue(result.has_multiple_reasons)
        self.assertGreaterEqual(
            result.reasons["bad_acting"]["explicit_evidence_count"], 1
        )
        self.assertGreater(
            result.reasons["bad_dialogue"]["heuristic_confidence"], 0
        )

    def test_public_single_review_interface_is_preserved(self) -> None:
        detector = NegativeReasonDetector()
        result = detector.analyze_review("The acting was terrible.")

        self.assertIn("bad_acting", result.labels)

    def test_batch_analysis_with_multiple_reviews(self) -> None:
        results = self.detector.analyze_reviews(
            [
                "The acting was terrible.",
                "The plot was weak and the ending was rushed.",
            ]
        )

        self.assertEqual(len(results), 2)
        self.assertIn("bad_acting", results[0].labels)
        self.assertIn("weak_plot_bad_writing", results[1].labels)
        self.assertIn("disappointing_ending", results[1].labels)
        self.assertTrue(results[1].has_multiple_reasons)

    def test_empty_review_inside_batch_analysis_returns_other_label(self) -> None:
        results = self.detector.analyze_reviews(["The dialogue was awful.", "", None])

        self.assertIn("bad_dialogue", results[0].labels)
        self.assertEqual(results[1].labels, [OTHER_LABEL])
        self.assertEqual(results[1].sentence_count, 0)
        self.assertEqual(results[2].labels, [OTHER_LABEL])
        self.assertEqual(results[2].sentence_count, 0)

    def test_to_dict_output_contains_export_friendly_fields(self) -> None:
        result = self.detector.analyze_review(
            "The acting was wooden and the dialogue was awful."
        )

        exported = result.to_dict()

        self.assertIn("labels", exported)
        self.assertIn("has_multiple_reasons", exported)
        self.assertIn("reason_scores", exported)
        self.assertIn("heuristic_confidence", exported)
        self.assertIn("supporting_sentences", exported)
        self.assertIn("matched_patterns", exported)
        self.assertNotIn("normalized_text", exported)
        self.assertIn("bad_acting", exported["reason_scores"])
        self.assertIn("bad_acting", exported["heuristic_confidence"])
        self.assertGreater(exported["heuristic_confidence"]["bad_acting"], 0)
        self.assertIn("bad_acting", exported["supporting_sentences"])
        self.assertIn("bad_acting", exported["matched_patterns"])

    def test_normalized_text_is_optional_in_result_and_export(self) -> None:
        compact_result = self.detector.analyze_review("The acting was terrible.")
        verbose_result = self.detector.analyze_review(
            "The acting was terrible.",
            include_normalized_text=True,
        )

        self.assertIsNone(compact_result.normalized_text)
        self.assertNotIn("normalized_text", compact_result.to_dict())
        self.assertEqual(verbose_result.normalized_text, "The acting was terrible.")
        self.assertEqual(
            verbose_result.to_dict(include_normalized_text=True)["normalized_text"],
            "The acting was terrible.",
        )

    def test_false_negative_expectations_and_writing_regression(self) -> None:
        result = self.detector.analyze_review(
            "I was really looking forward to this. The premise seemed promising, "
            "but the third act was wretched and heavy-handed."
        )

        self.assertNotEqual(result.labels, [OTHER_LABEL])
        self.assertIn("failed_expectations", result.labels)
        self.assertTrue(
            {"weak_plot_bad_writing", "disappointing_ending"}.intersection(
                result.labels
            )
        )

    def test_false_negative_story_dialogue_acting_and_visuals_regression(self) -> None:
        result = self.detector.analyze_review(
            "The story, or actually the lack thereof, was completely uninspired "
            "and lacked imagination. The dialogue and acting were even worse. "
            "The CGI effects look fake."
        )

        self.assertTrue(
            {
                "weak_plot_bad_writing",
                "bad_dialogue",
                "bad_acting",
                "poor_visuals_effects",
            }.issubset(set(result.labels))
        )

    def test_false_negative_boring_and_no_dialog_regression(self) -> None:
        result = self.detector.analyze_review(
            "This movie was extremely boring. I fell asleep three times and "
            "there was no dialog between characters."
        )

        self.assertNotEqual(result.labels, [OTHER_LABEL])
        self.assertIn("boring_slow_pacing", result.labels)
        self.assertIn("bad_dialogue", result.labels)

    def test_false_negative_plot_and_watchability_regression(self) -> None:
        result = self.detector.analyze_review(
            "This film has several key flaws, the most significant being the "
            "clear lack of a good plot. It was difficult to watch and felt "
            "like a waste of time."
        )

        self.assertNotEqual(result.labels, [OTHER_LABEL])
        self.assertIn("weak_plot_bad_writing", result.labels)
        self.assertIn("boring_slow_pacing", result.labels)

    def test_false_negative_expectations_originality_and_tone_regression(self) -> None:
        result = self.detector.analyze_review(
            "Perhaps my expectations were too high, because I was left a little dry. "
            "Where hasn't this setup been used before? Ultimately the film is too "
            "over-sexed to be a straight horror picture and too gruesome to work as "
            "a sex flick. There needs to be a balance."
        )

        self.assertTrue(
            {
                "failed_expectations",
                "generic_unoriginal",
                "tonal_mismatch",
            }.issubset(set(result.labels))
        )

    def test_false_negative_unclear_storyline_regression(self) -> None:
        result = self.detector.analyze_review(
            "No one could tell me what the actual storyline was. It was one of "
            "the most dreadful films I have seen."
        )

        self.assertIn("weak_plot_bad_writing", result.labels)

    def test_false_negative_direction_execution_and_writing_regression(self) -> None:
        result = self.detector.analyze_review(
            "The production choices are extremely poor and not fully realized. "
            "The screenplay should assist the development of the story, but the "
            "writer feels at a loss to where to go next. It is style over substance."
        )

        self.assertIn("poor_direction_execution", result.labels)
        self.assertIn("weak_plot_bad_writing", result.labels)

    def test_false_negative_weak_characters_regression(self) -> None:
        result = self.detector.analyze_review(
            "I began to lose faith halfway through. The main characters are "
            "obnoxious and less than relatable. I simply prefer brighter company."
        )

        self.assertIn("weak_characters", result.labels)

    def test_false_negative_factual_inaccuracy_regression(self) -> None:
        result = self.detector.analyze_review(
            "The film pretends to reenact real events, but many critical facts "
            "are demonstrably incorrect. There are too many gross errors and "
            "shameless inventions."
        )

        self.assertIn("factual_inaccuracy", result.labels)

    def test_not_funny_regression(self) -> None:
        result = self.detector.analyze_review(
            "This movie is not funny at all. Where was the joke?"
        )

        self.assertIn("not_funny", result.labels)

    def test_poor_production_quality_regression(self) -> None:
        result = self.detector.analyze_review(
            "The film has bad camera work, poor sound, and awful production values."
        )

        self.assertIn("poor_production_quality", result.labels)

    def test_expanded_visual_effects_regression(self) -> None:
        result = self.detector.analyze_review(
            "The special effects were terrible and the CGI looked fake."
        )

        self.assertIn("poor_visuals_effects", result.labels)

    def test_expanded_boring_pacing_regression(self) -> None:
        result = self.detector.analyze_review(
            "Do not waste your time. I fast forwarded through most of it."
        )

        self.assertIn("boring_slow_pacing", result.labels)

    def test_expanded_dialogue_and_screenplay_regression(self) -> None:
        result = self.detector.analyze_review(
            "The dialogue was atrocious and the screenplay was bad."
        )

        self.assertIn("bad_dialogue", result.labels)
        self.assertIn("weak_plot_bad_writing", result.labels)

    def test_expanded_bad_acting_regression(self) -> None:
        result = self.detector.analyze_review(
            "The acting is horrendous. None of these people can act."
        )

        self.assertIn("bad_acting", result.labels)

    def test_expanded_failed_expectations_regression(self) -> None:
        result = self.detector.analyze_review(
            "I wanted this movie to be good, but it was a huge disappointment."
        )

        self.assertIn("failed_expectations", result.labels)

    def test_expanded_tonal_mismatch_regression(self) -> None:
        result = self.detector.analyze_review(
            "It was supposed to be lighthearted, but it comes off as creepy."
        )

        self.assertIn("tonal_mismatch", result.labels)

    def test_sentence_splitting_handles_contrast_markers(self) -> None:
        result = self.detector.analyze_review(
            "The setup seemed promising; however the plot is stupid - bad editing too."
        )

        self.assertGreaterEqual(result.sentence_count, 3)
        self.assertIn("weak_plot_bad_writing", result.labels)
        self.assertIn("poor_production_quality", result.labels)


if __name__ == "__main__":
    unittest.main()
