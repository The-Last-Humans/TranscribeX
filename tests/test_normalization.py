import unittest

from transcribex.normalization import format_srt, normalize_funasr_result


class NormalizeFunASRResultTests(unittest.TestCase):
    def test_sentence_info_with_speakers_and_timestamps(self):
        raw = [
            {
                "text": "大家好我们开始今天的会议。好的。",
                "language": "zh",
                "sentence_info": [
                    {
                        "text": "大家好我们开始今天的会议。",
                        "start": 400,
                        "end": 3800,
                        "spk": 0,
                        "timestamp": [[400, 700], [700, 1000]],
                    },
                    {
                        "text": "好的。",
                        "start": 4200,
                        "end": 5100,
                        "spk": 1,
                    },
                ],
            }
        ]

        result = normalize_funasr_result(
            raw,
            model="paraformer-zh",
            duration=5.2,
            language=None,
            include_speakers=True,
            include_timestamps=True,
            speaker_map={"SPEAKER_00": "张三"},
        )

        self.assertEqual(result.language, "zh")
        self.assertEqual(result.duration, 5.2)
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].speaker, "张三")
        self.assertEqual(result.segments[0].start, 0.4)
        self.assertEqual(result.segments[0].end, 3.8)
        self.assertEqual(result.segments[1].speaker, "SPEAKER_01")

    def test_segments_merge_for_same_speaker_short_gap(self):
        raw = [
            {
                "sentence_info": [
                    {"text": "第一句。", "start": 0, "end": 1000, "spk": 0},
                    {"text": "第二句。", "start": 1200, "end": 2000, "spk": 0},
                    {"text": "第三句。", "start": 3000, "end": 4000, "spk": 1},
                ]
            }
        ]

        result = normalize_funasr_result(
            raw,
            model="paraformer-zh",
            duration=None,
            language="zh",
            include_speakers=True,
            include_timestamps=True,
        )

        self.assertEqual(len(result.segments), 2)
        self.assertEqual(result.segments[0].text, "第一句。第二句。")
        self.assertEqual(result.segments[0].end, 2.0)

    def test_srt_format(self):
        result = normalize_funasr_result(
            [{"text": "hello", "sentence_info": [{"text": "hello", "start": 1000, "end": 2500, "spk": 0}]}],
            model="paraformer-zh",
            duration=2.5,
            language="zh",
            include_speakers=True,
            include_timestamps=True,
        )

        self.assertIn("00:00:01,000 --> 00:00:02,500", format_srt(result.segments))
        self.assertIn("SPEAKER_00: hello", format_srt(result.segments))


if __name__ == "__main__":
    unittest.main()
