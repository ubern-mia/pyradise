from typing import Any


from pyradise.data.taping import Tape


class NewTape(Tape):
    def __init__(self):
        super().__init__()

    def record(self, value: Any) -> None:
        self.recordings.append(value)

    def playback(data: Any, **kwargs) -> Any:
        return data


def test_record_1():
    tape = NewTape()
    tape.record(10)
    assert len(tape.recordings) == 1
    assert tape.recordings[0] == 10


def test_record_2():
    tape = NewTape()
    tape.record(10)
    tape.record(22)
    assert len(tape.recordings) == 2
    assert tape.recordings[0] == 10
    assert tape.recordings[1] == 22


def test_get_recorded_elements_1():
    tape = NewTape()
    tape.record(10)
    tape.record(22)
    recorded_elements = tape.get_recorded_elements()
    assert len(recorded_elements) == 2
    assert recorded_elements[0] == 10


def test_get_recorded_elements_2():
    tape = NewTape()
    tape.record(10)
    tape.record(22)
    recorded_elements = tape.get_recorded_elements(reverse=True)
    assert len(recorded_elements) == 2
    assert recorded_elements[0] == 22


def test_reset():
    tape = NewTape()
    tape.record(10)
    tape.record(22)
    tape.reset()
    assert len(tape.recordings) == 0
