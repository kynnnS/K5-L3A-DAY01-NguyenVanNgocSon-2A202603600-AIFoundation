# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603600
- **Họ và tên:** Nguyễn Văn Ngọc Sơn
- **SEED** (in ở ô đầu notebook): 1420

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |


## 2. Kết quả chính

Provider LLM: Gemini (`gemini-3.5-flash-lite` là model nhỏ, `gemini-3.8-flash` là model lớn).

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.888 | 0.824 | 0.25 | 0 |
| 2. BoW + MLP | 0.944 | 0.919 | 2.06 | 0 |
| 3a. Encoder MiniLM + kNN | 0.909 | 1.000 | 71.83 | 0 |
| 3b. Encoder MiniLM + LogReg | 0.935 | 0.947 | 60.70 | 0 |
| 3c. Encoder MiniLM + MLP head | 0.941 | 0.923 | 44.82 | 0 |
| 4a. LLM nhỏ zero-shot | — | 1.000 | 844.03 | 0.031 |
| 4b. LLM lớn zero-shot | — | 1.000 | 3168.15 | 0.075 |
| 4c. LLM nhỏ few-shot | — | 1.000 | 897.52 | 0.077 |

Hàng LLM chỉ chạy trên SUB (40 tin) nên chỉ có cột F1 SUB 40. Với 40 tin, mỗi tin sai làm F1 đổi khoảng 2–3 điểm, nên chênh lệch dưới ~0.05 coi như hoà.

- Trên đường Pareto: 1, 2, 3c, 3b, 3a.
- Đạt SLA p95 < 50 ms: 1, 2, 3c.

## 3. Dự đoán và thực tế

**Dự đoán 1: `llm_models`**
- **Em đoán:** model lớn F1 ≈ 0.95, model nhỏ ≈ 0.92, chênh khoảng 0.03. Model lớn đắt gấp 2–3 lần.
- **Thực tế:** cả hai model đều đạt F1 = 1.000 trên SUB. Chi phí model lớn gấp khoảng 2.4 lần ($0.075 so với $0.031 / 1k tin), đúng như em đoán.
- **Vì sao lệch:** em đánh giá thấp LLM hiện tại. Sau khi em thêm vào prompt định nghĩa "dịch vụ nội dung trả phí là spam", hai tin trước đó bị sai được sửa. Ngoài ra, SUB chỉ có 40 tin nên rất dễ đạt 100%. Kết quả này không chứng minh LLM hoàn hảo.

**Dự đoán 2: `hard_eval`**
- **Em đoán:**
  - BoW + LogReg đạt khoảng 60%, LLM nhỏ khoảng 80%.
  - BoW khó nhất ở nhóm né từ khoá.
  - LLM khó nhất ở nhóm prompt injection.
- **Thực tế:**
  - BoW + LogReg chỉ đạt 50%. Nhóm khó nhất với nó là spam giọng tự nhiên (0%) và thao túng (12%), còn né từ khoá đạt 17%.
  - LLM nhỏ đạt 100% ở mọi nhóm, kể cả prompt injection. Cả prompt thường (4a) lẫn prompt phòng thủ (4d) đều 100%.
- **Vì sao lệch:** em nghĩ BoW chỉ gãy khi từ bị biến dạng. Thực ra BoW gãy nặng hơn khi tin spam *không có từ khoá nào*. Những tin viết giọng tự nhiên chứa nhiều từ như "i", "my", "me", và đây lại là các từ có trọng số kéo về ham. Về LLM, model đời mới đã được huấn luyện chống injection kiểu đơn giản như "Ignore all previous instructions".

**Dự đoán 3: `tradeoff`**
- **Em đoán:** BoW + LogReg và Embedding + LogReg (3b) nằm trên Pareto và đạt SLA. Gemini lớn cũng có thể nằm trên Pareto.
- **Thực tế:**
  - 3b có p95 = 60.7 ms, vượt SLA 50 ms. Mô hình đạt SLA với F1 cao nhất là 3c (44.8 ms).
  - Cả hai LLM đều không nằm trên Pareto, vì 3a đã đạt 1 − F1 = 0 trên SUB với độ trễ chỉ 71.8 ms.
- **Vì sao lệch:** em quên rằng p95 đo **từ văn bản thô**, tức là đã gồm thời gian encode bằng MiniLM trên CPU Colab. Phần chậm là bước encode, không phải đầu LogReg. Em cũng chưa tính đến việc trên tập 40 tin, nhiều cách cùng "chạm trần" F1 = 1.0, nên cách nhanh hơn sẽ lấn át cách chậm.

## 4. Phân tích 50 case khó

| Model | F1 SUB (test thường) | F1 50 case khó | Mức tụt |
| --- | --- | --- | --- |
| 1. BoW + LogReg | 0.824 | 0.286 | **−0.54** |
| 2. BoW + MLP | 0.919 | 0.432 | −0.49 |
| 3a. Emb + kNN | 1.000 | 0.704 | −0.30 |
| 3b. Emb + LogReg | 0.947 | 0.553 | −0.39 |
| 3c. Emb + MLP head | 0.923 | 0.565 | −0.36 |
| 4a / 4d. LLM nhỏ | 1.000 | 1.000 | 0 |

**Model tụt nhiều nhất là BoW + LogReg**, ở hai nhóm:
- **Spam giọng tự nhiên:** đúng 0/10.
- **Thao túng / lừa đảo:** đúng 12%.

Ngược lại, BoW làm tốt hai nhóm chỉ có ham (100%). Lý do là nó gần như chỉ đoán "ham" khi không thấy từ khoá spam. Tập test thường quá lạc quan vì lấy cùng phân phối với train. Spam mẫu chỉ khác số điện thoại vẫn có thể rơi vào cả train lẫn test, nên model chỉ cần nhớ từ khoá là đạt điểm cao.

**Case 1: #14** *"Hi, I'm working from home and making about £300 a day, happy to show you how, no experience needed, just text me back"* (spam, BoW đoán ham)
- Tin không chứa từ nào trong nhóm đẩy về spam như call, txt, free, www, claim.
- Ngược lại, nó có nhiều đại từ ngôi thứ nhất và từ hội thoại ("I'm", "me", "you"). Đây là các từ có trọng số âm mạnh, kéo về ham.
- BoW cộng trọng số từng từ độc lập, nên không thể nhận ra *ý định* rủ rê việc làm lừa đảo.

**Case 2: #11** *"Hey it's Jess from the gym 😊 … 3 months for the price of 1, just reply JOIN and I'll sort it"* (spam, BoW đoán ham)
- Giọng thân mật ("Hey it's Jess") kéo về ham.
- Tín hiệu khuyến mãi lại nằm ở *cụm từ* "3 months for the price of 1", không nằm ở một từ khoá đơn lẻ. Bag-of-words nhị phân không biểu diễn được thứ tự hay tổ hợp từ.
- Embedding và LLM nhìn được nghĩa cả câu, nên nhận ra đây là quảng cáo.

Không có case nào làm *tất cả* model cùng sai, vì LLM đúng 50/50.

## 5. Thử thách cá nhân

**Đề #5 (Embedding):** viết 4 tin nhắn gồm 2 spam né từ khoá và 2 ham chứa "free"/"win", rồi so sánh dự đoán của BoW với Embedding + LogReg.

**Cách làm:**
- Tự viết 4 tin, vector hoá bằng `vectorize` cho BoW và bằng `emb_model.encode` (chuẩn hoá `mu`/`sd`) cho Embedding.
- Dự đoán bằng trọng số đã train ở Phần 1 và Phần 3b, với ngưỡng 0.5.

| # | Tin | Thật | BoW | Emb + LogReg |
| --- | --- | --- | --- | --- |
| 1 | C0ngr4ts! Cl4im y0ur £750 g1ft v0ucher t0day. Repl¥ YES. | spam | ham ✗ | spam ✓ |
| 2 | Y0ur p4rcel is h3ld. P4y £1.95 at del1very-help.co t0day. | spam | ham ✗ | spam ✓ |
| 3 | Are you free after work? Let's play badminton at 6pm. | ham | ham ✓ | ham ✓ |
| 4 | We win the school quiz today! Let's celebrate tonight. | ham | ham ✓ | ham ✓ |

**Kết quả:** BoW đúng 2/4, Embedding + LogReg đúng 4/4.

**Giải thích:**
- **Tin 1 và 2:** thay chữ bằng số và ký hiệu ("C0ngr4ts", "Cl4im", "Repl¥") làm các token không còn trong vocab 2000 từ. BoW gần như mất hết tín hiệu spam, chỉ còn từ trung tính, nên đoán ham. MiniLM tách từ thành sub-word, vẫn giữ được các mảnh như "cl", "aim", "£", "voucher". Ngữ cảnh "trả phí để nhận hàng / nhận quà" đưa vector câu về vùng spam.
- **Tin 3 và 4:** "free" và "win" có trọng số đẩy về spam, nhưng câu còn nhiều từ hội thoại kéo về ham ("you", "let's", "today"). Vì vậy BoW vẫn đúng. Embedding đúng vì hiểu "free" = rảnh và "win" = thắng cuộc thi.

**Kết luận:** embedding vượt trội rõ ở spam né từ khoá. Với ham chứa từ khoá, cả hai đều đúng trên 4 tin này, nhưng cần nhiều tin hơn mới kết luận chắc chắn.

## 6. Phương án cho sản phẩm thật

**Yêu cầu:** 10 triệu SMS/ngày (≈ 116 tin/s trung bình, giờ cao điểm có thể tới ~600–1.200 tin/s), p95 < 50 ms/tin.

**Cách nào đạt độ trễ** (theo bảng của em):

| Cách | p95 | F1 test | F1 50 case khó |
| --- | --- | --- | --- |
| BoW + LogReg | 0.25 ms | 0.888 | 0.286 |
| BoW + MLP | 2.06 ms | 0.944 | 0.432 |
| Emb + MLP head | 44.82 ms | 0.941 | 0.565 |

- Emb + MLP head đạt SLA nhưng sát ngưỡng, chỉ đo trên CPU Colab.
- LLM không đạt: p95 từ 844 ms (model nhỏ) đến 3.168 ms (model lớn).
- **Chi phí nếu dùng LLM cho mọi tin:**
  - Model nhỏ: 10.000 × $0.031 ≈ **$310/ngày, ≈ $9.300/tháng**.
  - Model lớn: 10.000 × $0.075 ≈ **$750/ngày, ≈ $22.500/tháng**.
  - Đây là ước lượng thô, vì token được đếm theo số từ / 0.75.

**Phương án chọn: router 2 tầng.**
1. **Tầng 1: BoW + MLP xử lý 100% tin.**
   - p95 2 ms, F1 test 0.944.
   - Throughput đo được khoảng 25.758 tin/s cho cả lô, dư sức cho giờ cao điểm.
   - Không tốn phí API.
2. **Tầng 2: chỉ xử lý tin "lưỡng lự"** (p(spam) ∈ [0.3, 0.7]).
   - Chuyển sang Emb + MLP head, hoặc LLM nhỏ chạy **bất đồng bộ**, ngoài đường găng.
   - Tin được giao trước, nếu phát hiện spam thì gắn cờ hoặc thu hồi sau.
   - Nếu chỉ khoảng 5% tin vào tầng 2, chi phí LLM còn khoảng $15/ngày.
3. **Bổ sung:**
   - Chuẩn hoá đầu vào (leetspeak, ký tự ¥ € | → chữ cái) trước tầng 1 để giảm lỗi né từ khoá.
   - Định kỳ gom các tin bị người dùng báo cáo để huấn luyện lại.

**Đánh đổi chấp nhận:**
- Tầng 1 yếu ở spam giọng tự nhiên và lừa đảo mạo danh (F1 50 case khó chỉ 0.432). Một phần spam loại này sẽ lọt qua nếu model rất tự tin (p < 0.3).
- Phải chấp nhận độ trễ kiểm tra bổ sung *sau khi giao tin* cho nhóm lưỡng lự, đổi lại vẫn giữ được p95 < 50 ms ở đường chính.
- Cần đo lại p95 trên máy chủ thật, vì số đo hiện tại phụ thuộc CPU/GPU Colab.

## 7. AI log và điều đã học

**Công cụ AI đã dùng:** Claude (trợ lý chat) để sửa lỗi gọi API ở Phần 4 và review notebook.
**Nguyên tắc em tự đặt ra:** Phần 1–3 và thử thách cá nhân không nhờ AI viết code; mọi dự đoán `predict(...)` và các ô "Giải thích" đều do em tự viết.

| # | Phần | Em hỏi AI gì (prompt) | AI trả lời / viết gì | Em đã sửa / kiểm chứng gì |
|---|---|---|---|---|
| 1 | 4.1 `call_openai` (bản đầu) | *(nếu có)* "Viết hàm `call_openai` gọi Chat Completions qua endpoint OpenAI-compatible của Gemini, trả về (text, latency), dùng `**provider_kwargs(max_tokens)`" | Hàm tạo `OpenAI()` client, gọi `chat.completions.create(model, messages, temperature, top_p, **provider_kwargs(max_tokens))`, đo thời gian bằng `time.perf_counter()` | Chạy ô "Say hello in Vietnamese" → Gemini trả lời được (0.68s); đọc lại để hiểu `provider_kwargs` gửi `max_tokens=512` + `reasoning_effort` cho Gemini |
| 2 | 4.1 `call_openai` (bản thích ứng Gemini) | "Gemini báo lỗi 400 liên quan đến *thinking* khi gửi `reasoning_effort='minimal'`, sửa thế nào mà vẫn qua được test mock của OpenAI?" | Viết lại `call_openai`: với Gemini thử lần lượt `minimal → low → medium → high → bỏ hẳn tham số`; bắt `BadRequestError` có chữ "thinking" thì thử mức kế tiếp; lưu mức chạy được vào dict `GEMINI_EFFORT` để không thử lại; provider khác giữ nguyên 1 lần gọi | Chạy test mock 4.1 + 4.4 (provider giả lập = openai) → ✅; khi chạy 4.A thấy log `↳ [gemini-3.8-flash] dùng reasoning_effort = low`; thêm comment đầu ô đánh dấu ô do AI hỗ trợ; hiểu rằng model lớn không chấp nhận mức "minimal" |
| 3 | 4.2 Prompt phân loại | "Viết prompt zero-shot/few-shot phân loại SMS spam/ham, kết thúc bằng `Message: \"\"\"...\"\"\"` rồi `Label:`" | Prompt định nghĩa spam (quảng cáo, lừa đảo, phishing, trúng thưởng, dịch vụ nội dung trả phí như tin thể thao / nhạc chuông / chat), ham (tin cá nhân, giao dịch, tin chuyển tiếp giữa bạn bè); yêu cầu trả đúng 1 từ viết thường; few-shot chèn các cặp Message/Label | Chạy test 4.2–4.3 → ✅; em thêm vào định nghĩa spam mục "dịch vụ trả phí" sau khi thấy 2 tin bị đoán sai ở 4.A (tin tỉ số bóng đá Arsenal, tin "valentine game"); bỏ hàm `wrap()` không dùng vì mâu thuẫn với định dạng `"""` mà test yêu cầu; Run all để 4.A/4.B dùng đúng prompt cuối |
| 4 | 4.3 `parse_label` | *(nếu có)* "Viết hàm đọc nhãn spam/ham từ câu trả lời LLM, xử lý 'Not spam', dấu câu, chữ hoa, câu mơ hồ" | Chuẩn hoá về chữ thường chỉ giữ a–z; có "not spam" → 0; chỉ có "spam" → 1; chỉ có "ham" → 0; có cả hai hoặc không có → `None` | Chạy test với 7 case (`" Spam."`, `"Label: spam"`, `"HAM\n"`, `"Not spam"`, `"I am not sure"`, ...) → ✅; tự trả lời Bắt lỗi 3 về vì sao `"spam" in response` sai |
| 5 | 4.4 `llm_classify_batch` | *(nếu có)* "Viết hàm phân loại cả loạt, có retry khi bị 429, ước lượng chi phí, phù hợp gói Gemini miễn phí ~15 request/phút" | Vòng lặp gọi `call_llm` qua `retry_with_backoff(max_retries=5, base_delay=2.0)`; nhãn `None` → đếm `n_invalid` và gán 0; token ≈ số từ / 0.75; nghỉ 4s **ngoài** phần đo latency | Chạy test mock (kiểm tra cả `prompt_fn`) → ✅; kiểm tra p50/p95 của 4a (~655 ms / ~980 ms) không bị cộng 4s nghỉ; ghi nhận chi phí chỉ là ước lượng vì đếm token theo số từ |
| 6 | Review lần 1 | "Check lại notebook, những gì cần sửa và chưa xong, chỉ cách chỉnh sửa chi tiết" | Chỉ ra: (a) `metrics()` trả 0 cho cả accuracy khi mẫu số = 0, sai yêu cầu; (b) cosine kiểm tra vector 0 bằng tích vô hướng thay vì norm; (c) Câu 2.A ghi epoch 6–8 trong khi output là 9; (d) kết quả 4.A/4.B chạy trước khi sửa prompt; (e) các phần còn trống: stress test, tính tay, thử thách, phản ánh, AI log | Đối chiếu từng điểm với output trong notebook và thấy đúng; ghi danh sách việc cần làm |
| 7 | Câu 2.A: số epoch | "Ở phần 2.A epoch là 6–8 mà" (kèm ô test 2.2) | Giải thích ô em gửi là test 2.2, không phải 2.A; con số đúng nằm ở output "epoch có val loss thấp nhất: 9" (`np.argmin` trên val loss); gợi ý in train/val loss từng epoch để kiểm tra | In bảng loss từng epoch để tự xem; sửa Câu 2.A thành dừng ở epoch 9; sửa lại tên check `"2.2 train_mlp/"` (thừa dấu `/`) về đúng tên gốc |
| 8 | Câu 2.A ý 2 | "Trả lời lại phần 2.A cho em" | Không viết lại dự đoán đã ghi (vì đã có dấu thời gian trước khi chạy); đưa bảng số liệu (LogReg F1 0.888, MLP 0.944, bỏ ReLU 0.95, số tham số, thời gian train) và một khung câu có chỗ trống | Em tự điền kết luận: phần F1 tăng chủ yếu do cách train (Adam, mini-batch, nhiều epoch) chứ không phải do lớp ẩn, vì bản bỏ ReLU (mô hình tuyến tính) vẫn đạt 0.95 |
| 9 | Dự đoán `hard_eval` | "Dự đoán accuracy từng model trên 50 case và nhóm khó nhất" | AI **không viết dự đoán thay**; đưa câu hỏi gợi mở theo cách mỗi model nhìn văn bản (token rời rạc / vector nghĩa / đọc như người); sau khi em viết, nhắc rằng ô stress test chỉ chạy 4a và 4d, không có LLM lớn / few-shot | Em tự viết dự đoán trước khi chạy (có dấu thời gian); giữ nguyên nội dung dự đoán |
| 10 | Review lần 2 | "Check lại" (gửi notebook sau khi sửa) | Chỉ ra: 4.A vẫn chạy với prompt cũ; few-shot chạy lại ra F1 = 1.000 nhưng bảng tổng hợp và Câu 4 vẫn ghi 0.950; ô stress test không còn output; Câu 6 chưa dùng số liệu thật; `MY_CASES` thêm sau ô chạy nên chưa được chấm | Run all lại toàn bộ; viết lại Câu 4 ý 1 và Câu 5 theo số mới; chạy lại stress test sau khi thêm `MY_CASES`; viết lại Câu 6 dựa trên bảng % theo nhóm |

**Điều em học được khi dùng AI:** AI sửa nhanh được lỗi tích hợp API (dòng 2), nhưng code AI viết vẫn phải chạy test và đọc hiểu từng dòng. Review của AI giúp em phát hiện các kết quả bị lệch do chạy ô sai thứ tự (dòng 11), một lỗi em không tự thấy được.

**Điều em hiểu ra hôm nay:**
- Model lớn và đắt tiền chưa chắc là lựa chọn tốt nhất. Phải đặt chất lượng, độ trễ p95 và chi phí lên cùng một biểu đồ Pareto mới chọn đúng. Trong bài này, BoW + MLP tự train 128 nghìn tham số mới là phương án phù hợp nhất cho đường chính.
- Điểm cao trên tập test cùng phân phối có thể rất lạc quan. Chỉ khi chạy 50 case khó, em mới thấy BoW gần như không bắt được spam viết giọng tự nhiên.
- Cuối cùng, code do AI viết vẫn phải chạy test và đọc hiểu từng dòng. Thứ tự chạy ô cũng ảnh hưởng tới kết quả, nên cần "Run all" trước khi chép số liệu.
