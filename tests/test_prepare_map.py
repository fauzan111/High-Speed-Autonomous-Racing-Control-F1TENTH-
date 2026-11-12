import pytest

from scripts.prepare_map import prepare_map_assets


def test_prepare_map_copies_legacy_yaml_and_preserves_existing_config(tmp_path):
    track = tmp_path / "Spielberg"
    track.mkdir()
    legacy = track / "Spielberg.yaml"
    legacy.write_text("image: Spielberg.png\n")
    (track / "Spielberg_raceline.csv").touch()
    assert prepare_map_assets(track, "Spielberg") == track
    expected = track / "Spielberg_map.yaml"
    assert expected.read_text() == legacy.read_text()
    expected.write_text("image: custom.png\n")
    prepare_map_assets(track, "Spielberg")
    assert expected.read_text() == "image: custom.png\n"


def test_prepare_map_rejects_partial_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="raceline.csv"):
        prepare_map_assets(tmp_path, "Spielberg")
