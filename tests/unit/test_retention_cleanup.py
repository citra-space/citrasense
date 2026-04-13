"""Unit tests for retention cleanup and settings migration."""

import os
import time

from citrascope.analysis.retention import cleanup_previews, cleanup_processing_output
from citrascope.settings.citrascope_settings import CitraScopeSettings


class TestRetentionMigration:
    """Test the processing_output_retention_hours validator handles legacy bool values."""

    def test_true_migrates_to_keep_forever(self):
        s = CitraScopeSettings.model_validate({"processing_output_retention_hours": True})
        assert s.processing_output_retention_hours == -1

    def test_false_migrates_to_delete_immediately(self):
        s = CitraScopeSettings.model_validate({"processing_output_retention_hours": False})
        assert s.processing_output_retention_hours == 0

    def test_positive_int_preserved(self):
        s = CitraScopeSettings.model_validate({"processing_output_retention_hours": 24})
        assert s.processing_output_retention_hours == 24

    def test_negative_one_preserved(self):
        s = CitraScopeSettings.model_validate({"processing_output_retention_hours": -1})
        assert s.processing_output_retention_hours == -1

    def test_below_negative_one_clamped(self):
        s = CitraScopeSettings.model_validate({"processing_output_retention_hours": -5})
        assert s.processing_output_retention_hours == -1

    def test_default_is_zero(self):
        s = CitraScopeSettings.model_validate({})
        assert s.processing_output_retention_hours == 0

    def test_invalid_falls_back_to_zero(self):
        s = CitraScopeSettings.model_validate({"processing_output_retention_hours": "bogus"})
        assert s.processing_output_retention_hours == 0


class TestCleanupProcessingOutput:
    def test_deletes_old_dirs(self, tmp_path):
        old_dir = tmp_path / "task-old"
        old_dir.mkdir()
        (old_dir / "file.txt").write_text("x")
        # Backdate mtime to 48 hours ago
        old_time = time.time() - 48 * 3600
        os.utime(old_dir, (old_time, old_time))

        new_dir = tmp_path / "task-new"
        new_dir.mkdir()
        (new_dir / "file.txt").write_text("x")

        removed = cleanup_processing_output(tmp_path, retention_hours=24)
        assert removed == 1
        assert not old_dir.exists()
        assert new_dir.exists()

    def test_zero_retention_does_nothing(self, tmp_path):
        d = tmp_path / "task-1"
        d.mkdir()
        assert cleanup_processing_output(tmp_path, retention_hours=0) == 0
        assert d.exists()

    def test_negative_one_keeps_forever(self, tmp_path):
        d = tmp_path / "task-1"
        d.mkdir()
        old_time = time.time() - 9999 * 3600
        os.utime(d, (old_time, old_time))
        assert cleanup_processing_output(tmp_path, retention_hours=-1) == 0
        assert d.exists()

    def test_nonexistent_dir(self, tmp_path):
        assert cleanup_processing_output(tmp_path / "nope", retention_hours=24) == 0


class TestCleanupPreviews:
    def test_deletes_old_files(self, tmp_path):
        old_file = tmp_path / "old.jpg"
        old_file.write_text("x")
        old_time = time.time() - 60 * 86400
        os.utime(old_file, (old_time, old_time))

        new_file = tmp_path / "new.jpg"
        new_file.write_text("x")

        removed = cleanup_previews(tmp_path, retention_days=30)
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_nonexistent_dir(self, tmp_path):
        assert cleanup_previews(tmp_path / "nope") == 0
