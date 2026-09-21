from autocull_gui import _LogStream, app_control_hint, parse_progress


class _FakeSignal:
    def __init__(self):
        self.emitted = []

    def emit(self, text):
        self.emitted.append(text)


class TestParseProgress:
    def test_parses_tqdm_line(self):
        line = "Analyzing:  38%|###8      | 188/507 [00:22<00:31,  9.93it/s]"
        assert parse_progress(line) == ("Analyzing 188/507  00:22<00:31", 38)

    def test_ignores_plain_log_line(self):
        assert parse_progress("Found 507 images") is None

    def test_ignores_summary_line(self):
        assert parse_progress("Done - kept: 120, rejected: 387") is None


class TestLogStream:
    def test_carriage_return_goes_to_progress_only(self):
        log, progress = _FakeSignal(), _FakeSignal()
        stream = _LogStream(log, progress)
        stream.write("Analyzing:   0%| | 0/507 [00:00<?, ?it/s]\r")
        stream.write("Analyzing:  38%|#| 188/507 [00:22<00:31,  9.93it/s]\r")
        assert log.emitted == []
        assert len(progress.emitted) == 2

    def test_newline_goes_to_log(self):
        log, progress = _FakeSignal(), _FakeSignal()
        stream = _LogStream(log, progress)
        stream.write("Found 507 images\n")
        assert log.emitted == ["Found 507 images"]
        assert progress.emitted == []

    def test_progress_does_not_accumulate_into_one_line(self):
        log, progress = _FakeSignal(), _FakeSignal()
        stream = _LogStream(log, progress)
        for i in range(200):
            stream.write(f"\rAnalyzing:  {i // 2}%|#| {i}/507 [00:0{i % 10}<00:31,  9.93it/s]")
        assert log.emitted == []
        assert all(len(line) < 120 for line in progress.emitted)


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
