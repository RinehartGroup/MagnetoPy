import magnetopy as mp


class TestQDFile:
    def test_attrs(self, mvsh1: mp.QDFile):
        assert str(mvsh1) == 'File: "mvsh1.dat"'
        assert len(mvsh1.header) == 38
        assert mvsh1.sample_info.mass == 1.38
        assert mvsh1.data.shape == (229, 89)
        assert mvsh1.comments == ["MvsH, 20 C"]
        assert mvsh1.raw_header is None
        assert mvsh1.raw_data is None
        assert mvsh1.parsed_data.shape == (229, 20)
        assert mvsh1.measurement_type == "dc_dc"

    def test_raw_attrs(self, mvsh1_with_raw: mp.QDFile):
        assert len(mvsh1_with_raw.raw_header) == 28
        assert mvsh1_with_raw.raw_data.shape == (138316, 7)
        assert mvsh1_with_raw.parsed_data.shape == (229, 21)
