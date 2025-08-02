# MBTI & 얼굴 노화 예측 프로그램 (Python Version)

이 프로그램은 지정된 폴더(`input_images`)에 이미지를 추가하면, 해당 이미지를 분석하여 두 가지 결과를 `outputs` 폴더에 생성합니다.

1.  **MBTI 예측**: Teachable Machine으로 학습된 Keras 모델(`keras_model.h5`)을 기반으로 이미지의 MBTI를 예측하고, 관련 정보를 텍스트 파일로 저장합니다.
2.  **20년 뒤 얼굴 예측**: AI LAB API를 사용하여 20년 뒤의 얼굴을 예측하고, 결과 이미지를 저장합니다.

이 프로그램은 Python으로 작성되었으며, 윈도우 환경에서 실행되는 것을 기준으로 합니다.

## 사전 준비 (윈도우 기준)

1.  **Python 설치**:
    -   [Python 공식 웹사이트](https://www.python.org/downloads/windows/)에서 최신 버전의 Python을 다운로드하여 설치합니다.
    -   설치 과정에서 **"Add Python to PATH"** 또는 **"PATH에 Python 추가"** 옵션을 반드시 체크해야 합니다.

2.  **프로젝트 파일**:
    -   이 프로그램의 모든 파일 (`main.py`, `requirements.txt` 등)이 하나의 폴더에 있어야 합니다.
    -   `model` 폴더 안에 `keras_model.h5`와 `labels.txt` 파일이 있어야 합니다.
    -   프로젝트 폴더에 `db.json` 파일이 있어야 합니다.

## 설치 방법

1.  **CMD 또는 PowerShell 실행**:
    -   `Win + R` 키를 누르고 `cmd` 또는 `powershell`을 입력하여 명령 프롬프트를 엽니다.

2.  **프로젝트 폴더로 이동**:
    -   `cd` 명령어를 사용하여 이 프로그램이 있는 폴더로 이동합니다.
        ```sh
        cd C:\path\to\your\project\mbti_interface
        ```

3.  **필요한 라이브러리 설치**:
    -   아래 명령어를 실행하여 `requirements.txt`에 명시된 모든 라이브러리를 한번에 설치합니다.
        ```sh
        pip install -r requirements.txt
        ```
    -   설치에 다소 시간이 걸릴 수 있습니다.

## 사용법

1.  **프로그램 실행**:
    -   위와 동일한 명령 프롬프트 창에서 아래 명령어를 입력하여 파일 감시를 시작합니다.
        ```sh
        python main.py
        ```
    -   "Watching for new images in..." 메시지가 나타나면 프로그램이 정상적으로 실행된 것입니다. **이 창을 닫지 마세요.**

2.  **이미지 추가**:
    -   탐색기를 열어 `input_images` 폴더에 분석하고 싶은 얼굴 이미지 (`.jpg`, `.png` 등)를 복사하거나 이동시킵니다.

3.  **결과 확인**:
    -   프로그램이 이미지를 자동으로 감지하고 분석을 시작합니다. (터미널에 로그가 표시됩니다.)
    -   분석이 완료되면 `outputs` 폴더에 결과 파일이 생성됩니다.
        -   `[파일명]_mbti.txt`: MBTI 분석 결과 텍스트 파일
        -   `[파일명]_aged.jpg`: 20년 뒤 얼굴 예측 이미지

## 주의사항

-   `AILABAPI_KEY`는 `.env` 파일에 저장되어 있습니다. 키가 변경될 경우 이 파일을 수정해주세요.
-   `db.json` 파일이 없거나 내용에 오류가 있으면 프로그램이 시작되지 않습니다.
-   입력 이미지에는 분석 가능한 얼굴이 명확하게 포함되어야 API가 정상 동작합니다.
-   프로그램을 종료하려면 실행 중인 터미널 창에서 `Ctrl + C`를 누르세요.
