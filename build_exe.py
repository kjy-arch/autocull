"""
AutoCull 실행 파일 빌드 스크립트 (PyInstaller)

사용법:
    pip install pyinstaller
    python build_exe.py

결과: dist/AutoCull/ 폴더 (AutoCull.exe 포함)
다른 PC에서는 이 폴더 전체를 복사해서 AutoCull.exe 실행
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",           # 폴더 방식 (onefile보다 빠르고 안정적)
    "--windowed",         # 콘솔 창 없음
    "--name", "AutoCull",
    "--icon", str(HERE / "autocull.ico"),
    # 소스 파일들을 함께 패키징
    "--add-data", f"{HERE / 'autocull.py'};.",
    "--add-data", f"{HERE / 'grouper.py'};.",
    "--add-data", f"{HERE / 'analyzer.py'};.",
    "--add-data", f"{HERE / 'location.py'};.",
    # mediapipe 모델 파일
    "--collect-data", "mediapipe",
    # torch / open_clip hidden imports
    "--hidden-import", "torch",
    "--hidden-import", "open_clip",
    "--hidden-import", "timm",
    str(HERE / "autocull_gui.py"),
]

print("빌드 시작...")
print("(torch/mediapipe 포함으로 수 분 소요될 수 있습니다)\n")

result = subprocess.run(cmd, cwd=str(HERE))
if result.returncode == 0:
    print(f"\n✅ 빌드 완료: {HERE / 'dist' / 'AutoCull' / 'AutoCull.exe'}")
    print("dist/AutoCull/ 폴더 전체를 다른 PC에 복사해서 사용하세요.")
else:
    print("\n❌ 빌드 실패. 위 오류 메시지를 확인하세요.")
    sys.exit(1)
