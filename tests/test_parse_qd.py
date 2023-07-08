import magnetopy as mp


class TestQDFile:
    def test_attrs(self, mvsh5: mp.QDFile):
        assert str(mvsh5) == 'File: "mvsh5.dat"'
        assert len(mvsh5.header) == 38
        assert mvsh5.sample_info.mass == 1.38
        assert mvsh5.data.shape == (229, 89)
        assert mvsh5.comments == ["MvsH, 20 C"]
        assert mvsh5.raw_header is None
        assert mvsh5.raw_data is None
        assert mvsh5.parsed_data.shape == (229, 20)
        assert mvsh5.measurement_type == "dc_dc"

    def test_raw_attrs(self, mvsh5_with_raw: mp.QDFile):
        assert len(mvsh5_with_raw.raw_header) == 28
        assert mvsh5_with_raw.raw_data.shape == (138316, 7)
        assert mvsh5_with_raw.parsed_data.shape == (229, 21)
