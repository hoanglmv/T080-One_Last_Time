# Commit Guideline

Tài liệu này quy định cách các thành viên làm việc với Git trong dự án
`P080-One_Last_Time`.

## Quy trình nhanh

Mỗi lần thực hiện một task, thành viên làm theo thứ tự sau:

```text
1. Chuyển sang branch cá nhân
2. Đồng bộ code mới nhất từ develop
3. Thay đổi code và chạy kiểm tra
4. Chỉ stage các file liên quan
5. Commit với message đúng quy ước
6. Push branch cá nhân
7. Tạo Pull Request vào develop
```

Không commit hoặc push trực tiếp lên `main` và `develop` khi chưa được trưởng
nhóm thống nhất.

## Cấu trúc branch

```text
main
└── develop
    ├── hoanglmv
    ├── tuelv
    ├── thangnt
    └── phongnv
```

- `main`: phiên bản ổn định, sẵn sàng demo hoặc deploy.
- `develop`: nơi tích hợp và kiểm thử tính năng của cả nhóm.
- Branch cá nhân: nơi mỗi thành viên phát triển và commit hằng ngày.

Không push code trực tiếp vào `main`. Thay đổi phải đi theo luồng:

```text
branch cá nhân → Pull Request → develop → Pull Request → main
```

Branch cá nhân hiện có của nhóm:

| Thành viên | Branch |
|------------|--------|
| Hoàng | `hoanglmv` |
| Tuệ | `tuelv` |
| Thắng | `thangnt` |
| Phong | `phongnv` |

## Bắt đầu làm việc

Chuyển sang branch cá nhân:

```bash
git switch <ten-branch>
```

Ví dụ:

```bash
git switch hoanglmv
```

Cập nhật code mới nhất từ `develop` trước khi bắt đầu:

```bash
git fetch origin
git merge origin/develop
```

Không dùng `git pull origin develop` khi đang đứng nhầm branch. Luôn kiểm tra
branch hiện tại trước:

```bash
git branch --show-current
```

Nếu phát sinh conflict, không chọn nội dung một cách máy móc. Đọc cả hai phía,
trao đổi với người sửa cùng file khi cần, sau đó chạy lại toàn bộ kiểm tra.

## Kiểm tra thay đổi

Xem trạng thái repository:

```bash
git status
```

Xem nội dung đã thay đổi:

```bash
git diff
```

Chạy kiểm tra trước khi commit:

```bash
uv sync --frozen
make check
```

Nếu máy không có `make`:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

## Tạo commit

Chỉ stage các file liên quan đến thay đổi:

```bash
git add path/to/file1 path/to/file2
```

Không dùng `git add .` hoặc `git add -A` nếu chưa kiểm tra kỹ `git status`, vì
có thể vô tình đưa `.env`, file log hoặc thay đổi của task khác vào commit.

Kiểm tra lại nội dung chuẩn bị commit:

```bash
git diff --staged
```

Tạo commit:

```bash
git commit -m "<type>: <mô tả ngắn>"
```

Mỗi commit chỉ nên giải quyết một mục đích cụ thể. Không gom nhiều thay đổi
không liên quan vào cùng một commit.

Commit message nên:

- dùng tiếng Anh để thống nhất với lịch sử repository;
- bắt đầu bằng động từ ở dạng hiện tại như `add`, `fix`, `update`, `remove`;
- mô tả kết quả thay đổi, không mô tả chung chung quá trình làm;
- không thêm dấu chấm ở cuối tiêu đề;
- giữ dòng tiêu đề ngắn gọn, nên dưới 72 ký tự.

## Quy tắc commit message

Sử dụng định dạng:

```text
<type>: <mô tả ngắn>
```

Các `type` thường dùng:

| Type | Khi sử dụng |
|------|-------------|
| `feat` | Thêm tính năng mới |
| `fix` | Sửa lỗi |
| `docs` | Thay đổi tài liệu |
| `test` | Thêm hoặc sửa test |
| `refactor` | Cải tổ code nhưng không đổi hành vi |
| `perf` | Cải thiện hiệu năng |
| `style` | Format hoặc thay đổi không ảnh hưởng logic |
| `build` | Thay đổi dependencies hoặc build system |
| `ci` | Thay đổi CI/CD |
| `chore` | Công việc bảo trì khác |

Ví dụ tốt:

```text
feat: add chat history endpoint
fix: handle empty user message
docs: update uv setup instructions
test: add coverage for agent routing
refactor: extract OpenAI client configuration
build: add chromadb dependency
```

Nếu thay đổi cần giải thích thêm, dùng phần body:

```bash
git commit -m "fix: preserve chat history on retry" \
  -m "Keep the previous messages when the LLM request is retried after a timeout."
```

Tránh các commit message không rõ nội dung:

```text
update
fix
change code
done
commit mới
```

## Push branch cá nhân

```bash
git push origin <ten-branch>
```

Ví dụ:

```bash
git push origin hoanglmv
```

Nếu đây là lần push đầu tiên của branch:

```bash
git push -u origin <ten-branch>
```

Sau khi thiết lập tracking, các lần tiếp theo chỉ cần:

```bash
git push
```

Lệnh push sẽ kích hoạt AI Log pre-push hook. Nếu thành viên chưa cài hook, chạy:

```bash
bash scripts/setup_hooks.sh
```

Thành viên Windows PowerShell chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_hooks.ps1
```

## Tạo Pull Request vào `develop`

Trên GitHub, tạo Pull Request với:

```text
base: develop
compare: <ten-branch>
```

Pull Request cần có:

- tiêu đề mô tả rõ thay đổi;
- tóm tắt những gì đã thực hiện;
- cách kiểm thử;
- ảnh chụp hoặc video nếu thay đổi giao diện;
- ghi chú về API key, migration hoặc cấu hình mới nếu có.

Mẫu nội dung Pull Request:

```markdown
## Thay đổi

- Mô tả ngắn các thay đổi chính

## Kiểm thử

- [ ] `uv run ruff check src tests`
- [ ] `uv run mypy src`
- [ ] `uv run pytest`

## Lưu ý

- Migration, biến môi trường hoặc công việc còn lại (nếu có)
```

Chỉ merge khi:

- các kiểm tra local đã pass;
- CI chạy thành công nếu workflow được kích hoạt cho Pull Request;
- không còn conflict với `develop`;
- thay đổi đã được review.

Ưu tiên **Squash and merge** nếu branch có nhiều commit nhỏ hoặc commit sửa lỗi
lặp lại. Dùng merge thông thường khi cần giữ từng commit có ý nghĩa độc lập.

Sau khi merge, cập nhật lại branch cá nhân:

```bash
git switch <ten-branch>
git fetch origin
git merge origin/develop
```

## Đưa `develop` vào `main`

Chỉ thực hiện khi phiên bản trên `develop` đã được tích hợp và kiểm thử ổn định.
Tạo Pull Request:

```text
base: main
compare: develop
```

Không merge branch cá nhân trực tiếp vào `main`.

## Sửa sai trước khi push

Nếu vừa commit nhưng thiếu file hoặc sai commit message và **chưa push**, có thể
sửa commit gần nhất:

```bash
git add path/to/missing-file
git commit --amend
```

Nếu chỉ cần sửa message:

```bash
git commit --amend -m "fix: correct commit message"
```

Không amend hoặc force-push commit đã được người khác sử dụng. Nếu thay đổi đã
push, ưu tiên tạo commit sửa lỗi mới. Không chạy `git push --force` trên
`develop` hoặc `main`.

Nếu stage nhầm file nhưng chưa commit, bỏ file khỏi staging area mà vẫn giữ nội
dung trên máy:

```bash
git restore --staged path/to/file
```

## Những điều không được commit

Không commit:

- `.env` hoặc API key;
- `.venv`;
- mật khẩu, token hoặc credential;
- file cache như `__pycache__`, `.pytest_cache`, `.ruff_cache`;
- dữ liệu cá nhân hoặc dữ liệu production;
- file lớn không cần thiết.

Trước mỗi commit, luôn chạy:

```bash
git status
git diff --staged
```

Nếu phát hiện credential đã được commit, không chỉ xóa file ở commit tiếp theo.
Hãy thu hồi key ngay và thông báo cho trưởng nhóm để xử lý lịch sử Git.
