from pyradise.data.taping import TransformInfo, TransformTape


def test_record():
    tra_tape = TransformTape()
    tra_info = TransformInfo("name_1", None, None, None, None, None, None)
    tra_tape.record(tra_info)
    tra_info = TransformInfo("name_2", None, None, None, None, None, None)
    tra_tape.record(tra_info)
    assert tra_tape.recordings[0].name == "name_1"
    assert tra_tape.recordings[1].name == "name_2"


def test_get_recorded_elements_1():
    tra_tape = TransformTape()
    tra_info = TransformInfo("name_1", None, None, None, None, None, None)
    tra_tape.record(tra_info)
    tra_info = TransformInfo("name_2", None, None, None, None, None, None)
    tra_tape.record(tra_info)
    recordings = tra_tape.get_recorded_elements()
    assert recordings[0].name == "name_1"
    assert recordings[1].name == "name_2"


def test_get_recorded_elements_2():
    tra_tape = TransformTape()
    tra_info = TransformInfo("name_1", None, None, None, None, None, None)
    tra_tape.record(tra_info)
    tra_info = TransformInfo("name_2", None, None, None, None, None, None)
    tra_tape.record(tra_info)
    recordings = tra_tape.get_recorded_elements(reverse=True)
    assert recordings[0].name == "name_2"
    assert recordings[1].name == "name_1"
