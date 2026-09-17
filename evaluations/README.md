# Bộ đánh giá RAG

Thư mục này chứa nhãn đánh giá có phiên bản cho Study Assistant. Corpus hiện tại chỉ gồm
`Nhóm3_TTCSN.docx`, là tài liệu do chủ dự án cho phép sử dụng. File DOCX gốc không được commit;
hãy lập chỉ mục file đó trong workspace local trước khi chạy benchmark.

## Quy tắc gán nhãn

- `relevant_sources` dùng đúng nhãn nguồn sau bước chunking, ví dụ `Đoạn 25–34` hoặc `Bảng 1`.
- `expected_facts` là các cụm dữ kiện tối thiểu cần xuất hiện trong câu trả lời, không phải đáp án
  mẫu để chép nguyên văn.
- Ca `answerable: false` không có nguồn hay dữ kiện kỳ vọng và phải dẫn đến từ chối.
- `required_terms` ghi mã, tên riêng hoặc thuật ngữ không được làm mất khi biến đổi truy vấn.
- `development` dùng để điều chỉnh cấu hình; không điều chỉnh theo kết quả `holdout`.
- Mỗi thay đổi nhãn phải tăng `dataset_version` hoặc `corpus_version` tương ứng.

## Chạy dense baseline

Đóng Streamlit trước khi mở cùng kho Qdrant local, đặt API key trong terminal rồi chạy:

```bash
export GEMINI_API_KEY="..."
uv run python -m scripts.evaluate_rag \
  --dataset evaluations/datasets/lapzone-v1.json \
  --split holdout
```

Kết quả máy đọc và báo cáo ngắn được ghi vào `evaluations/results/`. Không commit API key,
nội dung DOCX gốc hoặc output chứa dữ liệu riêng ngoài corpus đã được duyệt.

## Cách đọc kết quả

- Hit Rate@K: tỷ lệ ca có ít nhất một nguồn đúng trong top K.
- Recall@K: tỷ lệ nguồn đúng được tìm thấy.
- MRR: nguồn đúng đầu tiên xuất hiện sớm đến mức nào.
- Citation precision/recall/coverage: trích dẫn có đúng và đủ nguồn chuẩn không.
- Expected-fact coverage: tỷ lệ dữ kiện tối thiểu xuất hiện trong câu trả lời.
- Answerability F1: chất lượng quyết định trả lời hay từ chối.

Corpus một tài liệu phù hợp để dựng baseline đầu tiên nhưng chưa đại diện cho truy xuất xuyên tài
liệu. Giới hạn này phải được nêu khi công bố kết quả.

Benchmark sẽ dừng thay vì xuất report nếu hash DOCX hoặc fingerprint ingestion khác nhãn đã
phiên bản hóa. Khi thay đổi corpus, embedding, chunking hoặc contextualization, hãy tạo dataset
version mới và gán nhãn lại; schema hiện tại chỉ hỗ trợ một tài liệu đã xác thực hash.
