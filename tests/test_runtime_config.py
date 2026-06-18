import tempfile
import unittest
from pathlib import Path

from transcribex.config import Settings
from transcribex.profiles import DeviceFacts, GPUInfo, recommend_profiles
from transcribex.runtime_config import RuntimeConfigPatch, RuntimeConfigStore


class RuntimeConfigTests(unittest.TestCase):
    def test_default_config_requires_setup_when_no_config_file_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                config_path=Path(tmp) / "config.json",
                require_setup=True,
                asr_model="paraformer-zh",
                device="cpu",
            )
            store = RuntimeConfigStore(settings)

            config = store.current()

        self.assertFalse(config.setup_complete)
        self.assertEqual(config.asr_model, "paraformer-zh")
        self.assertEqual(config.device, "cpu")

    def test_apply_profile_persists_runtime_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(config_path=Path(tmp) / "config.json")
            store = RuntimeConfigStore(settings)

            applied = store.apply(RuntimeConfigPatch(profile_id="cpu-multilingual"))
            loaded = store.current()

        self.assertTrue(applied.setup_complete)
        self.assertEqual(loaded.profile_id, "cpu-multilingual")
        self.assertEqual(loaded.asr_model, "iic/SenseVoiceSmall")
        self.assertEqual(loaded.spk_model, "cam++")


class RecommendationTests(unittest.TestCase):
    def test_cpu_recommendation_prefers_balanced(self):
        facts = DeviceFacts(os="Linux", machine="x86_64", cpu_count=8, memory_gb=32, nvidia_gpus=[])

        recommended = recommend_profiles(facts)

        self.assertEqual(recommended[0].id, "cpu-balanced")

    def test_large_gpu_includes_accuracy_profile(self):
        facts = DeviceFacts(
            os="Linux",
            machine="x86_64",
            cpu_count=16,
            memory_gb=64,
            nvidia_gpus=[GPUInfo(name="RTX", memory_mb=24_000)],
        )

        recommended = recommend_profiles(facts)

        self.assertIn("gpu-accuracy", [profile.id for profile in recommended])


if __name__ == "__main__":
    unittest.main()
