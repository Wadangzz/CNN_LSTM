# CNN-LSTM 기반 진동 예측 모델
![1](https://github.com/user-attachments/assets/dedc5694-d381-4404-91d0-8e9383612063)
![2](https://github.com/user-attachments/assets/1f7b1437-cb3b-4d55-bc12-30908cb3af24)

**CNN-LSTM 시계열 예측 모델을 활용하여 진동 데이터를 분석하고**
**미세한 이상 진동 패턴을 조기에 감지**함으로써 **예지보전(PdM: Predictive Maintenance)**에 활용하는 것을 목표로 합니다.

> 🔄 이 프로젝트는 현재 개발 중이며, 구조 개선 및 기능 추가가 지속적으로 이루어질 예정입니다.

---

## 🚧 현재 진행 중인 작업

- 다양한 주파수와 진폭이 혼합된 **합성 진동 데이터 생성**
- CNN-LSTM 구조를 통해 시계열 진동 데이터를 학습
- 단일 또는 다중 시점 예측 (single-step / multi-step)
- 예측된 데이터와 실제 데이터를 비교하여 **FFT(고속 푸리에 변환)** 주파수 성분 분석 수행

---

## 📦 구성 모듈

* `CNN_LSTM.py` – fft 손실함수, CNN-LSTM 모델 (Pytorch)

* `test.py` – json 데이터 read, 학습 데이터 전처리, 학습 후 결과 Plot

---
## 🔗 License

MIT License. Free to use, modify, and learn from.

---
