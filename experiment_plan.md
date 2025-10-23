## 🤖 머신_러닝_성능_최적화를_위한_실험_계획서.md

제시해주신 7가지 핵심 요소를 바탕으로, 최종 모델을 도출하기 위한 체계적인 실험 계획서를 작성했습니다.

실험은 **'폭포수(Waterfall)' 방식과 '그리드 탐색(Grid Search)'의 조합**으로 접근합니다. 모든 조합을 테스트하는 것은 비효율적이므로(Combinatorial Explosion), 각 단계(Phase)에서 최적의 요소를 선별하고, 이 선별된 요소를 다음 단계의 기본값(Baseline)으로 활용하여 실험을 누적해나가는 방식을 제안합니다.

---

### 1. 실험 목표

- 5종의 한국어 PLM(Pre-trained Language Model) 성능 비교
- 데이터 전처리, 증강, 손실 함수, 튜닝 기법(DAPT, LoRA) 등 다양한 요소가 모델 성능에 미치는 영향 분석
- 앙상블 기법을 포함한 최적의 성능을 내는 **최종 모델 파이프라인(Final Model Pipeline)** 확립

---

### 2. 베이스라인(Baseline) 설정

모든 비교 실험의 출발점(Control Group)이 될 기본 설정을 정의합니다.

- **Model:** `klue/bert-base` (가장 표준적인 모델 중 하나)
- **Preprocessing:** 모델의 기본 Tokenizer 사용 (특별한 정제 작업 X)
- **Data Augmentation:** 적용 안 함 (Ratio: 0%)
- **Loss Function:** Cross-Entropy Loss
- **Training:** Full Fine-tuning (LoRA, DAPT 미적용)
- **Ensemble:** 적용 안 함

---

### 3. 실험 설계 및 절차

#### Phase 1: 핵심 백본 모델(Backbone Model) 선정

가장 큰 성능 차이를 유발하는 기본 모델 아키텍처를 먼저 비교합니다.

- **실험 1.1: 5종 PLM 성능 비교**
  - **목표:** 동일한 조건 하에서 가장 높은 잠재력을 보이는 모델 선정.
  - **변수 (Models):**

    1. `klue/roberta-base`(후보1)
    2. `klue/bert-base`
    3. `kykim/bert-kor-base`(Baseline)
    4. `beomi/kcbert-base`
    5. `monologg/koelectra-base-v3-discriminator`
  - **통제 변인:** **Baseline 설정**의 모든 요소 (기본 전처리, CE Loss, Full-tuning 등)

    실험 결과
  - ![](assets/20251023_141218_image.png)

---

#### Phase 2: 데이터 최적화 (Data Optimization)

선정된 `Best_Model`을 기준으로, 모델에 입력되는 데이터의 품질을 높이는 실험을 진행합니다.

- **실험 2.1: 전처리(Preprocessing) 기법 비교**

  - **목표:** 노이즈 제거 및 텍스트 정제가 성능에 미치는 영향 확인.
  - **기준 모델:** `Best_Model`
  - **변수 (Methods):**

    1. Baseline (기본 Tokenizer만)
    2. 특수문자, 이모티콘 등 노이즈 제거
    3. 불필요한 공백 제거 및 오탈자 교정 (선택적)
    4. 2 + 3 (병합)
  - **결과:** 가장 성능이 우수한 전처리 방식(이하 **`Best_Preprocess`**)을 선정합니다.

    > **1023 현재 베이스 코드**
    >

    ```
    # 텍스트 전처리 파이프라인 클래스 구성
    class TextPreprocessingPipeline:
        """
        텍스트 전처리 파이프라인 클래스
        - 기본 전처리와 학습 데이터 기반 고급 전처리를 통합 관리
        - 재사용 가능하고 확장 가능한 구조
        """

        def __init__(self):
            self.is_fitted = False
            self.vocab_info = {}
            self.label_patterns = {}

        def basic_preprocess(self, texts):
            """기본 전처리 (clean_text + normalize 기능)"""
            processed_texts = []
            for text in texts:
                # 기본 텍스트 정리
                cleaned = self._clean_text(text)
                processed_texts.append(cleaned)
            return processed_texts

        def _clean_text(self, text):
            """기존 clean_text 함수 내용"""
            if pd.isna(text):
                return ""

            text = str(text).strip()
            text = text.lower() # 소문자 변환
            text = self._remove_urls_emails_mentions(text) # URL, 이메일, 멘션 제거
            text = self._normalize_punctuation(text)  # 구두점 정규화
            #text = self._remove_incomplete_korean(text)
            text = self._normalize_emotion_expressions(text) # 감정 표현 정규화 (ㅋㅋㅋ , ㅎㅎㅎ)
            text = self._reduce_excessive_repetition(text) # 과도한 문자 반복 축소 (아아아아아아앙 -> 아아아아)
            text = self._clean_special_characters(text) # 특수문자 제거 (이모티콘, 특수기호)
            text = self._normalize_whitespace(text) # 공백 정규화 (여러 개의 공백 -> 하나의 공백)

            return text.strip()

        def fit(self, texts, labels=None):
            """학습 데이터로부터 전처리 정보 학습 (품질 검사 기준 학습)"""

            self.is_fitted = True
            print("✓ 전처리 파이프라인 학습 완료")


        def transform(self, texts):
            """전처리 적용 (품질 문제 데이터 제거 + 텍스트 전처리)"""
            if not self.is_fitted:
                print(
                    "Warning: 파이프라인이 학습되지 않았습니다. 기본 전처리만 적용합니다."
                )
                return self.basic_preprocess(texts)

            # 텍스트 전처리 적용
            return self.basic_preprocess(texts)

        def fit_transform(self, texts, labels=None):
            """학습과 변환을 동시에 수행"""
            # 1. 학습 단계 (품질 검사 기준 학습)
            self.fit(texts, labels)

            # 2. 변환 단계 (품질 문제 데이터 제거 + 텍스트 전처리)
            processed_texts = self.transform(texts)

            # 3. 라벨도 동일하게 필터링
            return processed_texts


        @staticmethod
        def _remove_incomplete_korean(text):
            """불완전한 한글 제거 (자음/모음만 있는 경우)"""
            return re.sub(r"[ㄱ-ㅎㅏ-ㅣ]+", "", text)

        @staticmethod
        def _normalize_emotion_expressions(text):
            """감정 표현 정규화"""
            def replace_emotion(match):
                char = match.group(1)
                count = len(match.group(0))
                # log2x + 1 공식을 정수로 변환
                new_count = int(math.log2(count)) + 1 if count > 0 else 1
                return char * new_count

            # 웃음과 슬픔 표현 정규화 (2번 이상 반복)
            text = re.sub(r"([ㅋㅎ])\1+", replace_emotion, text)
            text = re.sub(r"([ㅠㅜㅡ])\1+", replace_emotion, text)
            return text

        @staticmethod
        def _reduce_excessive_repetition(text):
            """과도한 문자 반복 축소 (4번 이상 → 3번으로)"""

            def replace_repetition(match):
                char = match.group(1)
                count = len(match.group(0))
                # log2x + 1 공식을 정수로 변환하고 최소 1개 보장
                new_count = max(1, int(math.log2(count)) + 1) if count > 0 else 1
                return char * new_count

            return re.sub(r"(.)\1{3,}", replace_repetition, text)

        @staticmethod
        def _clean_special_characters(text):
            """특수문자 제거 (이모티콘 보존)"""

            # 1. 허용할 이모티콘 범위 정의
            # emoji_ranges = r"\U0001F600-\U0001F64F"  # Emoticons
            # emoji_ranges += r"\U0001F300-\U0001F5FF"  # Misc Symbols/Pictographs
            # emoji_ranges += r"\U0001F680-\U0001F6FF"  # Transport/Map
            # emoji_ranges += r"\U00002600-\U000026FF"  # Misc Symbols (★ 포함)
            # emoji_ranges += r"\U00002700-\U000027BF"  # Dingbats

            # 2. 허용할 기타 특수기호 정의
            other_symbols = r"@★#$" # 예시로 @ 추가

            # 3. 허용할 문자들을 조합하여 정규식 생성
            #allowed_chars = rf"\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-{emoji_ranges}{other_symbols}"
            allowed_chars = rf"\w\s가-힣.,!?ㅋㅎㅠㅜㅡ~\-"

            return re.sub(rf"[^{allowed_chars}]", " ", text)


        @staticmethod
        def _normalize_whitespace(text):
            """공백 정규화"""
            return re.sub(r"\s+", " ", text)

        @staticmethod
        def _normalize_punctuation(text):
            """구두점 정규화 (log 공식 적용)"""
            def replace_punctuation(match):
                char = match.group(1)
                count = len(match.group(0))
                # log2x + 1 공식을 정수로 변환
                new_count = int(math.log2(count)) + 1 if count > 0 else 1
                return char * new_count

            # 각 구두점별로 log 공식 적용
            text = re.sub(r"([.])\1+", replace_punctuation, text)
            text = re.sub(r"([!])\1+", replace_punctuation, text)
            text = re.sub(r"([?])\1+", replace_punctuation, text)
            text = re.sub(r"([,])\1+", replace_punctuation, text)

            # 구두점 앞뒤 공백 정리
            text = re.sub(r"\s+([.,!?])", r"\1", text)
            text = re.sub(r"([.,!?])\s+", r"\1 ", text)

            return text

        @staticmethod
        def _remove_urls_emails_mentions(text):
            """URL, 이메일, 멘션 제거"""
            # URL 패턴 제거
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )
            # 이메일 패턴 제거
            text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "", text)
            # 멘션 패턴 제거
            text = re.sub(r"\B@\w+", "", text)

            return text

    ```



- **실험 2.2: 데이터 증강(Data Augmentation) 방식 및 비율 비교** - agent구성이 필요하여 대기

  - **목표:** 데이터 불균형 해소 및 일반화 성능 향상을 위한 최적의 증강 전략 탐색.
  - **기준 모델:** `Best_Model` + `Best_Preprocess`
  - **변수 (Methods):**
    1. 적용 안 함 (Baseline)
    2. Back-Translation (역번역)
    3. EDA (Easy Data Augmentation: Synonym Replacement, Random Deletion 등)
  - **변수 (Ratios):** 각 Method 별로 원본 대비 증강 데이터 비율 (예: 25%, 50%, 100%)을 테스트합니다.
  - **결과:** 최적의 증강 방식 및 비율(이하 **`Best_Aug`**)을 선정합니다.

---

#### Phase 3: 학습 파이프라인 고도화 (Advanced Training)

데이터가 준비된 상태에서, 모델의 학습 방식 자체를 고도화합니다.

- **실험 3.1: DAPT (Domain-Adaptive Pre-Training) 적용**

  - **목표:** 보유한 데이터(도메인)에 모델을 사전 적응시켜 성능 향상 확인.
  - **기준:** `Best_Model` + `Best_Preprocess` + `Best_Aug`
  - **변수:**
    1. DAPT 미적용 (Baseline)
    2. 보유한 Train (또는 Train+Test) 데이터의 Unlabeled Corpus를 활용하여 DAPT 선행 학습 진행
  - **결과:** DAPT 적용 여부(이하 **`Best_DAPT`**)를 결정합니다.
- **실험 3.2: 손실 함수(Loss Function) 비교**

  - **목표:** 클래스 불균형 등 데이터 특성에 맞는 손실 함수 탐색.
  - **기준:** `Best_Model` + `Best_Preprocess` + `Best_Aug` + `Best_DAPT`
  - **변수:**
    1. Cross-Entropy Loss (Baseline)
    2. **Focal Loss** (불균형 데이터에 유리)
    3. Label Smoothing (모델의 과신(Over-confidence) 방지)
  - **결과:** 최적의 손실 함수(이하 **`Best_Loss`**)를 선정합니다.
- **실험 3.3: 파인튜닝(Fine-tuning) 기법 비교**

  - **목표:** Full-tuning 대비 PEFT(LoRA)의 성능 및 효율성 비교.
  - **기준:** `Best_Model` + `Best_Preprocess` + `Best_Aug` + `Best_DAPT` + `Best_Loss`
  - **변수:**
    1. **Full Fine-tuning** (Baseline)
    2. **LoRA (Low-Rank Adaptation)** (Rank, Alpha 등 하이퍼파라미터 튜닝 필요)
  - **결과:** 성능과 학습/추론 효율을 고려하여 최적의 튜닝 방식(이하 **`Best_Tuning`**)을 결정합니다.

---

#### Phase 4: 앙상블(Ensemble) 및 최종 모델 결정

개별 모델의 성능을 극대화한 후, 여러 모델을 결합하여 추가 성능 향상을 도모합니다.

- **실험 4.1: 앙상블 전략 비교**

  - **목표:** 단일 모델의 한계를 극복하고 예측 안정성 확보.
  - **기준:** Phase 3까지의 최적 파이프라인으로 학습된 **'Best Single Model'**
  - **변수:**
    1. **(A) Heterogeneous Ensemble (서로 다른 모델):**
       - Phase 1에서 선정된 Top 2~3 모델 (예: `klue/roberta`, `koelectra`)을 각각 Phase 3까지의 최적 파이프라인으로 학습시킨 뒤, Soft/Hard Voting 앙상블.
    2. **(B) Homogeneous Ensemble (동일 모델):**
       - 'Best Single Model'을 K-Fold 교차 검증 시 생성된 K개의 모델 또는 서로 다른 Random Seed로 K번 학습한 모델들을 Soft/Hard Voting 앙상블.
  - **결과:** (A), (B) 전략과 'Best Single Model'의 성능을 비교합니다.
- **실험 4.2: 최종 모델(Final Model) 제시**

  - 실험 4.1의 결과(A 또는 B)가 'Best Single Model' 대비 유의미한 성능 향상(및 추론 시간 허용 범위 내)을 보인다면 앙상블 모델을 채택합니다.
  - 그렇지 않다면, Phase 3까지의 **'Best Single Model'**을 최종 모델로 확정합니다.

---

### 4. 평가 지표 (Evaluation Metrics)

실험의 성패를 판단하기 위한 지표입니다. (태스크에 따라 달라질 수 있음)

- **주요 지표 (Primary):** **Macro F1-Score** (클래스 불균형을 고려한 가장 일반적인 지표)
- **보조 지표 (Secondary):**
  - Accuracy
  - Precision, Recall (per-class)
  - Confusion Matrix (오분류 패턴 분석)
- **비용 지표 (Cost):**
  - 학습 시간 (Training Time)
  - 추론 속도 (Inference Speed, (예: samples/sec))
  - (LoRA 적용 시) 학습 가능한 파라미터 수
