# Cài đặt AI Log Hook cho thành viên

AI Log Hook ghi lại hoạt động sử dụng AI vào `.ai-log/session.jsonl` và gửi các
log chưa nộp lên hệ thống chấm khi thành viên chạy `git push`. Mỗi thành viên
chỉ cần cài hook một lần trên từng máy sau khi clone repository.

## 1. Yêu cầu

- Đã clone repository và đang đứng tại thư mục gốc của dự án.
- Có Git và Python 3.11 trở lên. Script tự tìm `python3`, `python` hoặc `py -3`.
- Có invitation key do ban tổ chức cấp. Không chia sẻ hoặc commit key này.
- Đã cấu hình email Git để log được gắn đúng thành viên:

```bash
git config user.email "email-cua-ban@example.com"
```

Kiểm tra lại bằng:

```bash
git config user.email
```

## 2. Cấu hình biến môi trường

Tạo file `.env` từ template nếu máy chưa có:

### Linux, macOS hoặc Git Bash

```bash
cp .env.example .env
```

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

Mở `.env` và cập nhật ba biến sau:

```dotenv
AI_LOG_SERVER=https://ai-logs.note.transformerlabs.ai/api/ingest
AI_LOG_API_KEY=invitation-key-cua-ban
AI_LOG_DIR=.ai-log
```

File `.env` đã được `.gitignore` loại trừ. Không dùng `git add -f .env` và không
đưa API key vào ảnh chụp màn hình, issue hoặc pull request.

## 3. Cài pre-push hook

> Script cài đặt quản lý `.git/hooks/pre-push`. Nếu bạn đã có pre-push hook
> riêng, hãy sao lưu hoặc tích hợp nội dung của hook đó trước khi chạy.

### Linux, macOS hoặc Git Bash

```bash
bash scripts/setup_hooks.sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_hooks.ps1
```

Khi thành công, terminal hiển thị:

```text
[ai-log] Git pre-push hook installed.
[ai-log] Setup complete. Configure AI_LOG_SERVER in your .env file.
```

Hook này chạy hai tác vụ trước mỗi lần push:

1. Thu thập prompt gần đây của Antigravity/Gemini nếu có.
2. Gửi các bản ghi đang chờ trong `.ai-log/session.jsonl`.

Lỗi logging hoặc lỗi mạng không chặn `git push`. Nếu gửi thất bại, log được giữ
lại trên máy và sẽ được thử lại ở lần push tiếp theo.

## 4. Hook theo công cụ AI

Các cấu hình đã nằm sẵn trong repository. Thành viên phải mở công cụ AI từ thư
mục dự án để công cụ đọc đúng cấu hình:

| Công cụ | Cấu hình | Cơ chế |
| --- | --- | --- |
| Claude Code | `.claude/settings.json` | Prompt, tool call và kết thúc phiên |
| OpenAI Codex CLI/app | `.codex/hooks.json` | Prompt và kết thúc lượt |
| Cursor | `.cursor/hooks.json` | Prompt và kết thúc lượt |
| Gemini CLI | `.gemini/settings.json` | Trước agent, sau model và kết thúc phiên |
| GitHub Copilot | `.github/hooks/hooks.json` | Prompt và kết thúc phiên |
| Antigravity IDE | `.agents/rules/ai-log-hook.md` | Quét transcript trước khi push |

Sau khi clone hoặc pull thay đổi cấu hình hook, hãy đóng và mở lại công cụ AI
nếu hook chưa được nhận diện.

### Bước bắt buộc riêng cho Codex

Codex yêu cầu người dùng xác nhận riêng từng command hook, kể cả khi project đã
được đánh dấu là trusted. Sau khi mở lại Codex trong thư mục dự án:

1. Chạy `/hooks`.
2. Xác nhận Codex đã tìm thấy `.codex/hooks.json`.
3. Chọn trust/enable cho `log-prompt` và `log-stop`.
4. Gửi một prompt mới rồi kiểm tra `.ai-log/session.jsonl`.

Codex lưu trust theo hash của hook. Vì vậy, nếu nội dung hook được cập nhật, bạn
có thể phải mở `/hooks` và trust lại.

## 5. Ghi log cho mọi AI không có hook

Không thể tự động đọc transcript của mọi website AI vì trình duyệt và từng dịch
vụ không cung cấp một hook chung cho repository. Với ChatGPT web, Claude.ai,
Gemini web, Perplexity hoặc bất kỳ công cụ không hỗ trợ hook, chạy chế độ tương
tác:

```bash
bash scripts/_pyrun.sh scripts/log_manual.py
```

Hoặc ghi nhanh bằng một lệnh:

```bash
bash scripts/_pyrun.sh scripts/log_manual.py \
  --tool chatgpt \
  --model gpt-5 \
  --prompt "Tóm tắt nội dung đã yêu cầu AI hỗ trợ"
```

Chỉ ghi mô tả công việc cần thiết. Không đưa password, API key, dữ liệu cá nhân
hoặc bí mật của dự án vào prompt hay phần kết quả.

## 6. Kiểm tra cài đặt

### Kiểm tra pre-push hook

Linux, macOS hoặc Git Bash:

```bash
test -x .git/hooks/pre-push && echo "AI Log pre-push hook is ready"
```

Windows PowerShell:

```powershell
Test-Path .git/hooks/pre-push
```

### Tạo một log thử

```bash
bash scripts/_pyrun.sh scripts/log_manual.py \
  --tool hook-test \
  --prompt "Kiểm tra cài đặt AI Log Hook"
```

Sau đó kiểm tra file `.ai-log/session.jsonl`. File này phải có thêm một dòng
JSON chứa đúng `repo`, `branch`, `commit` và email trong trường `student`.

Log thật được gửi tự động ở lần `git push` tiếp theo. Không cần commit file
`.ai-log/session.jsonl`; nội dung log đã được `.gitignore` loại trừ.

## 7. Xử lý lỗi thường gặp

- **Không tạo được log:** kiểm tra `python3 --version` hoặc `python --version`,
  sau đó mở lại AI tool từ thư mục gốc của repository.
- **`student` bị trống hoặc sai:** chạy lại `git config user.email`.
- **Push báo `AI_LOG_SERVER not set`:** kiểm tra file tên chính xác là `.env` và
  ba biến ở bước 2 không bị comment.
- **Server trả lỗi xác thực:** cập nhật `AI_LOG_API_KEY` bằng invitation key đúng
  của thành viên; không dùng key mẫu trong `.env.example`.
- **Gửi log thất bại do mạng:** không xóa `.ai-log/session.jsonl`; hook sẽ tự thử
  lại ở lần push tiếp theo.
- **Đã pull cấu hình mới nhưng hook không chạy:** chạy lại script ở bước 3 và
  khởi động lại AI tool.
- **Codex không log:** chạy `/hooks`, trust/enable hai hook của project rồi gửi
  một prompt mới. Nếu dùng Codex CLI trong WSL, chạy `codex --version` và cài
  lại CLI nếu package native Linux đang bị thiếu.

## Checklist cho thành viên mới

- [ ] Clone repository và checkout branch cá nhân.
- [ ] Cấu hình đúng `git config user.email`.
- [ ] Tạo `.env` và điền `AI_LOG_API_KEY` cá nhân.
- [ ] Chạy script cài pre-push hook phù hợp với hệ điều hành.
- [ ] Khởi động AI tool từ thư mục gốc dự án.
- [ ] Tạo một manual log thử và xác nhận `.ai-log/session.jsonl` có dữ liệu.
- [ ] Push branch và kiểm tra terminal có thông báo submit log.
