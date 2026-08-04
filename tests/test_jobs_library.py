"""Tests for taking recordings out of the list, and for the folder they came from.

Three operations that all look alike from the sidebar and are not alike at all on
disk. Archiving changes a flag. Deleting a job throws away work that costs minutes
of CPU to make again. Deleting the recording as well throws away something that
cannot be made again by anybody, which is why it is a separate word.

The tests below mostly pin the differences, because a swipe is a small gesture and
the wrong one behind it is expensive.
"""

from __future__ import annotations

import json

import pytest

from scriba import jobs


@pytest.fixture
def library(tmp_path, monkeypatch):
    """A jobs folder with one processed recording in it."""
    monkeypatch.setattr(jobs, "JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "jobs").mkdir()

    def make(name: str, **state) -> "object":
        job = tmp_path / "jobs" / name
        (job / "output").mkdir(parents=True)
        (job / "output" / "doc.md").write_text("# a document")
        (job / "audio16k.wav").write_bytes(b"\0" * 4096)
        source = tmp_path / f"{name}.m4a"
        source.write_bytes(b"\0" * 8192)
        (job / "state.json").write_text(json.dumps(
            {"source": str(source), "duration": 90.0,
             "names": {"SPEAKER_00": "Ada"}, **state}))
        return job

    return make


def test_a_new_job_is_not_archived(library):
    library("memo-aaa111")
    assert jobs.inventory()[0].archived is False


def test_archiving_hides_it_from_the_list_and_keeps_the_work(library):
    job = library("memo-aaa111")
    jobs.archive(job)

    row = jobs.inventory()[0]
    assert row.archived is True
    assert row.state == "done"                      # still finished
    assert (job / "output" / "doc.md").exists()     # still there
    # And the rest of the state survived the edit. It carries the names somebody
    # typed in and the fingerprints that decide whether the cache is still good.
    assert row.names == {"SPEAKER_00": "Ada"}


def test_archiving_can_be_undone(library):
    job = library("memo-aaa111")
    jobs.archive(job)
    jobs.archive(job, value=False)
    assert jobs.inventory()[0].archived is False


def test_archiving_a_job_with_unreadable_state_still_works(library, tmp_path):
    """A folder half-written by a killed run is exactly what somebody tidies up."""
    job = library("memo-aaa111")
    (job / "state.json").write_text("{ this is not json")
    jobs.archive(job)
    assert json.loads((job / "state.json").read_text())["archived"] is True


def test_forgetting_deletes_the_job_and_keeps_the_recording(library, tmp_path):
    job = library("memo-aaa111")
    source = tmp_path / "memo-aaa111.m4a"

    freed = jobs.forget(job)
    assert not job.exists()
    assert source.exists(), "the recording is not the tool's to delete by default"
    assert freed > 0


def test_forgetting_with_the_source_deletes_both(library, tmp_path):
    job = library("memo-aaa111")
    source = tmp_path / "memo-aaa111.m4a"

    jobs.forget(job, with_source=True)
    assert not job.exists()
    assert not source.exists()


def test_forgetting_a_job_whose_recording_has_moved_does_not_fail(library, tmp_path):
    job = library("memo-aaa111")
    (tmp_path / "memo-aaa111.m4a").unlink()
    jobs.forget(job, with_source=True)
    assert not job.exists()


def test_forgetting_something_that_is_not_there_frees_nothing(tmp_path):
    assert jobs.forget(tmp_path / "not-a-job") == 0.0


def test_the_collection_travels_with_the_job(library):
    library("memo-aaa111", collection="Interviews/March")
    assert jobs.inventory()[0].collection == "Interviews/March"


def test_a_job_with_no_collection_reports_an_empty_one(library):
    library("memo-aaa111")
    assert jobs.inventory()[0].collection == ""
