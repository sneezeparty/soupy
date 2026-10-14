"""tools/prune_flux_sources.py removes only exact copies of archived pictures that nothing points at."""

from __future__ import annotations

import json

from tools import prune_flux_sources as prune


def _media(tmp_path):
    images, sources = tmp_path / "images", tmp_path / "flux_sources"
    images.mkdir()
    sources.mkdir()
    (images / "1_cat.png").write_bytes(b"cat picture")
    (images / "2_dog.png").write_bytes(b"dog picture")
    (sources / "copy_of_cat.png").write_bytes(b"cat picture")
    (sources / "copy_of_dog.png").write_bytes(b"dog picture")
    (sources / "upload.png").write_bytes(b"a photo someone uploaded")
    (sources / "same_size.png").write_bytes(b"cat pictur3")  # same length as the cat, different bytes
    return tmp_path


def test_only_exact_copies_of_archived_pictures_are_listed(tmp_path):
    media = _media(tmp_path)
    assert [p.name for p in prune.find_duplicates(media)] == ["copy_of_cat.png", "copy_of_dog.png"]


def test_a_copy_that_a_result_was_made_from_is_kept(tmp_path):
    media = _media(tmp_path)
    rows = [
        {"filename": "3_x.png", "source_image": "flux_sources/copy_of_dog.png"},
        {"filename": "4_y.png", "source_image": "images/1_cat.png"},
        {"filename": "5_z.png"},
    ]
    (media / "images" / "index.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8"
    )
    assert [p.name for p in prune.find_duplicates(media)] == ["copy_of_cat.png"]


def test_nothing_is_removed_without_delete(tmp_path, capsys):
    media = _media(tmp_path)
    assert prune.main(["--media", str(media)]) == 0
    assert len(list((media / "flux_sources").iterdir())) == 4
    assert "2 of 4 files" in capsys.readouterr().out

    assert prune.main(["--media", str(media), "--delete"]) == 0
    assert sorted(p.name for p in (media / "flux_sources").iterdir()) == ["same_size.png", "upload.png"]
    assert sorted(p.name for p in (media / "images").iterdir()) == [
        "1_cat.png",
        "2_dog.png",
    ]  # the archive is untouched


def test_a_missing_folder_is_not_an_error(tmp_path):
    assert prune.main(["--media", str(tmp_path / "nowhere")]) == 0
    assert prune.find_duplicates(tmp_path) == []
