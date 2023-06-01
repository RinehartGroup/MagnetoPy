class TestQDFile:
    def test_attrs(self, mvsh1):
        assert str(mvsh1) == 'File: "mvsh1.dat"'
        assert len(mvsh1.header) == 38
        assert mvsh1.sample_info.mass == 1.38
