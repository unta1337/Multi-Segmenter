# Multi-Segmentor
한국기술교육대학교 2023학년도 1학기 멀티코어프로그래밍 중간 텀프로젝트.

# 빌드 방법
cmake와 Visual Studio 2022가 설치된 상태에서

```pwsh
cmake -G "Visual Studio 17 2022" -A x64 CMakeLists.txt -B x64
cmake --build x64 --target Multi-Segmenter Multi-Segmenter-Test --config Release
```

`x64\Release\Multi-Segmenter.exe` 경로에 실행 파일이 생성된다.

# 실행 방법

```pwsh
.\bin\x64\Multi-Segmenter.exe [Mode (serial or parallel)] [Tolerance (Float, Optional)] [ObjFilePath]
```