from autocull_gui import app_control_hint


class TestAppControlHint:
    def test_detects_korean_block_message(self):
        trace = (
            'Traceback (most recent call last):\n'
            '  File "autocull.py", line 14, in <module>\n'
            'ImportError: DLL load failed while importing cv2: '
            '애플리케이션 제어 정책에서 이 파일을 차단했습니다.\n'
        )
        hint = app_control_hint(trace)
        assert hint is not None
        assert "Smart App Control" in hint

    def test_detects_english_block_message(self):
        trace = "ImportError: DLL load failed while importing cv2: blocked by application control policy.\n"
        assert app_control_hint(trace) is not None

    def test_returns_none_for_unrelated_error(self):
        trace = "ImportError: DLL load failed while importing cv2: The specified module could not be found.\n"
        assert app_control_hint(trace) is None
