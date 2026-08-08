# Alternative Credit Scoring — Home Credit POC

POC end-to-end cho bài toán dự đoán `payment difficulty` trên Home Credit 2018:

- leakage-aware EDA và feature engineering dùng chung giữa training/serving;
- `LogisticRegression` baseline, `LightGBM` champion, early stopping và Platt calibration;
- ROC-AUC, PR-AUC, KS, Gini, Brier, ECE, PSI, temporal-stability adapter và fairness diagnostic;
- FastAPI + form web: nhập hồ sơ → score → risk band → local feature contributions;
- LLM explanation tùy chọn, chỉ diễn đạt reason codes và không tham gia tính điểm.

> Đây là POC nghiên cứu. `TARGET` là payment difficulty theo định nghĩa cuộc thi; score không phải CIC và không được dùng làm quyết định duyệt/từ chối tự động. Home Credit 2018 không có time key nên kết quả hiện tại không chứng minh temporal stability.

Tài liệu chính: [Implementation Plan](IMPLEMENTATION_PLAN.md), [Model Card](artifacts/models/MODEL_CARD.md), [training report](artifacts/models/training_report.json) và [EDA report](artifacts/reports/eda_report.json).

## 🐳 Chạy Dự Án Bằng Docker (1-Click Docker Setup)

Dự án hỗ trợ đóng gói và khởi chạy toàn bộ ứng dụng (FastAPI + Web Dashboard + Pre-trained Models) bằng Docker chỉ với **1 câu lệnh duy nhất**.

### 1. Cài đặt Docker (Nếu chưa có)

- **Trên Linux / WSL2 (Ubuntu)**:
  ```bash
  sudo apt update && sudo apt install -y docker.io docker-compose-v2
  sudo service docker start
  sudo usermod -aG docker $USER
  ```
- **Trên Windows / macOS**:
  Tải và mở [Docker Desktop](https://www.docker.com/products/docker-desktop/). Nếu dùng Windows WSL2, vào **Docker Desktop Settings -> Resources -> WSL integration** và bật tick chọn bản Ubuntu của bạn.

---

### 2. Khởi chạy 1-Click bằng Docker

```bash
docker compose up --build
```
*Hoặc dùng script 1-click:*
```bash
./run/run_docker.sh
```

Mở trình duyệt:
- 📊 **Web Dashboard**: <http://localhost:8000/api/v1/credit/demo>
- 📚 **Swagger API Docs**: <http://localhost:8000/docs>

---

## 💻 Chạy Trực Tiếp Bằng Local Python (Dành cho Developer)

```bash
uv sync --frozen

# Chỉ cần chạy lại nếu muốn tái tạo artifact từ raw data
uv run python -m src.credit_scoring.cli eda \
  --data-dir data/raw/home-credit-default-risk
uv run python -m src.credit_scoring.cli train \
  --data-dir data/raw/home-credit-default-risk \
  --feature-set serving

uv run uvicorn src.main:app --reload --port 8000
```

Mở:

- POC form: <http://localhost:8000/api/v1/credit/demo>
- Swagger: <http://localhost:8000/docs>
- Model metadata: <http://localhost:8000/api/v1/credit/model>
- Readiness: <http://localhost:8000/ready>

API mẫu:

```bash
curl -X POST http://localhost:8000/api/v1/credit/score \
  -H 'Content-Type: application/json' \
  -d '{
    "application": {
      "AMT_INCOME_TOTAL": 180000,
      "AMT_CREDIT": 450000,
      "AMT_ANNUITY": 27000,
      "AMT_GOODS_PRICE": 420000,
      "DAYS_BIRTH": -12775,
      "DAYS_EMPLOYED": -1825,
      "CNT_CHILDREN": 1,
      "CNT_FAM_MEMBERS": 3,
      "EXT_SOURCE_1": 0.55,
      "EXT_SOURCE_2": 0.62,
      "EXT_SOURCE_3": 0.48
    },
    "explain_with_llm": false,
    "top_k": 6
  }'
```

Để bật narration qua LLM, cấu hình `OPENAI_API_KEY`, đặt `ENABLE_LLM_EXPLANATIONS=true` và gửi `explain_with_llm=true`. Score và reason codes vẫn hoạt động nếu LLM lỗi hoặc bị tắt.

Kiểm tra project:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

---

## Hạ tầng template ban đầu

Template chính thức cho học viên **VinUni AI20K Build Phase** — cung cấp sẵn cấu trúc dự án, code mẫu, và hướng dẫn kỹ thuật chi tiết để xây dựng AI Agent đạt điểm cao (35+/50).

> 📖 **Technical Guidebook:** [phoenix.note.transformerlabs.ai/technical-book](https://phoenix.note.transformerlabs.ai/technical-book)

## 🎯 Template này dùng để làm gì?

Khi tham gia AI20K Build Phase, mỗi đội cần xây dựng một AI Agent hoàn chỉnh — từ kiến trúc, code, test, đến deploy. Thay vì bắt đầu từ con số không, template này cung cấp:

- **Cấu trúc thư mục chuẩn** — đã được thiết kế theo best practices (separation of concerns)
- **Code mẫu** cho các phần cốt lõi: LangGraph agent, FastAPI API, config, schemas
- **Docker + CI/CD sẵn** — Dockerfile multi-stage, GitHub Actions workflow
- **Hướng dẫn kỹ thuật 10 chương** — từ clone template đến nộp bài Demo Day
- **Checklist 10 deliverables** — đảm bảo không bỏ sót yêu cầu BTC
- **AI Usage Logging tự động** — Pre-configured hooks cho Claude Code, Cursor, Codex, Gemini CLI, Antigravity, và GitHub Copilot

## ⚡ Quick Start

Yêu cầu duy nhất là [`uv`](https://docs.astral.sh/uv/). Project khóa Python
và toàn bộ dependencies trong `.python-version` và `uv.lock`, vì vậy không cần
cài Python hay tạo virtual environment thủ công.

### 1. Clone project

```bash
git clone https://github.com/hoanglmv/P080-One_Last_Time.git
cd P080-One_Last_Time
```

### 2. Cài Astral `uv` (Linux)

```bash
# Installer chính thức của Astral
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Mở terminal mới để nạp lại `PATH`, sau đó kiểm tra:

```bash
uv --version
```

### 3. Cài Python và dependencies

```bash
uv sync --frozen
```

Lệnh này tự động:

- tải đúng Python 3.11 nếu máy chưa có;
- tạo `.venv`;
- cài chính xác dependencies từ `uv.lock`.

Không cần activate `.venv`. Hãy chạy các lệnh Python của project qua `uv run`.

### 4. Cấu hình biến môi trường (Linux)

```bash
cp .env.example .env
```

Mở `.env` và cập nhật ít nhất:

```dotenv
OPENAI_API_KEY=your-openai-api-key
AI_LOG_API_KEY=your-invitation-key
```

`AI_LOG_API_KEY` là key riêng trong link mời của BTC. Không commit `.env` hoặc
bất kỳ API key nào lên Git.

### 5. Cài AI Logging Hooks (Linux)

```bash
bash scripts/setup_hooks.sh
```

Chỉ cần cài hooks một lần sau khi clone.

Thành viên dùng Windows, cần cấu hình từng AI tool hoặc cần kiểm tra/xử lý lỗi,
xem [Hướng dẫn cài AI Log Hook cho thành viên](docs/guide/setup/ai-log-hooks.md).

### 6. Kiểm tra project

```bash
make check
```

Nếu máy không có `make`, chạy:

```bash
uv run ruff check src tests
uv run mypy src
uv run pytest
```

### 7. Chạy ứng dụng

```bash
uv run uvicorn src.main:app --reload --port 8000
```

Sau khi server khởi động:

- API root: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

Các lệnh thường dùng:

```bash
make run        # Chạy development server
make test       # Chạy tests
make lint       # Kiểm tra code style
make typecheck  # Kiểm tra type annotations
make check      # Chạy lint, type-check và tests
```

## 📁 Cấu trúc dự án

```
├── src/
│   ├── agents/           # 🧠 LangGraph Agent
│   │   ├── graph.py      #    State graph (nodes + edges)
│   │   ├── state.py      #    State schema (TypedDict)
│   │   ├── nodes/        #    Node functions
│   │   └── tools/        #    Agent tools (@tool)
│   ├── api/              # 🌐 FastAPI Backend
│   │   └── routes.py     #    API endpoints
│   ├── models/           # 📋 Pydantic schemas
│   ├── services/         # 🔧 Business logic (LLM, etc.)
│   ├── config.py         # ⚙️ Pydantic Settings
│   └── main.py           # 🚀 App entry point
├── tests/                # 🧪 pytest suite
│   ├── test_agents/      #    Agent/graph tests
│   └── test_api/         #    API endpoint tests
├── scripts/              # 🔌 AI Logging Hooks
│   ├── log_hook.py       #    Auto-log cho Claude/Cursor/Codex/Gemini/Copilot
│   ├── log_antigravity.py#    Antigravity IDE prompt scanner
│   ├── log_manual.py     #    Manual log cho ChatGPT / web tools
│   ├── submit_log.py     #    Submit logs on git push
│   └── setup_hooks.sh    #    One-time hook installer
├── .claude/ .codex/ .cursor/ .gemini/  # Per-tool hook configs
├── .agents/              # Antigravity rules + workflows
├── .ai-log/              # 📊 AI usage logs (auto-generated)
├── docs/
│   ├── guide/            # 📖 Technical Guidebook (10 chapters)
│   └── architecture_diagram.md
├── eval/                 # 📊 Evaluation results
├── presentation/         # 🎤 Demo Day slides
├── .github/workflows/    # ⚡ CI/CD (GitHub Actions)
├── .github/hooks/        # 🪝 Copilot hook config
├── Dockerfile            # 🐳 Multi-stage build
├── docker-compose.yml    # 🐙 Full stack orchestration
├── pyproject.toml        # 📦 Metadata và khai báo dependencies
├── uv.lock               # 🔒 Phiên bản dependency tái lập trên mọi máy
└── README_boilerplate.md # 📝 README template cho đội của bạn
```

## 📚 Technical Guidebook — 10 Chương

| Chương | Nội dung | Thời gian |
|---------|----------|-----------|
| 1 | Lời mở đầu — Mục tiêu, cách sử dụng | 15 phút |
| 2 | Khởi tạo dự án — Clone, setup, git workflow | 4 giờ |
| 3 | Thiết kế kiến trúc — 3-tier, diagrams, ADR | 6 giờ |
| 4 | **LangGraph Agent** — State, nodes, edges, tools, RAG | 8 giờ |
| 5 | FastAPI — Routes, validation, error handling, streaming | 6 giờ |
| 6 | Giao diện — Next.js + Streamlit quickstart | 6 giờ |
| 7 | DevOps — Docker, CI/CD, deploy, logging | 6 giờ |
| 8 | Kiểm thử — Unit test, integration test, RAGAS | 4 giờ |
| 9 | Demo Day — 10 deliverables, checklist, tips | 2 giờ |
| 10 | Tài nguyên — Khóa học, docs, BMAD method | tham khảo |

📖 **Đọc online:** [phoenix.note.transformerlabs.ai/technical-book](https://phoenix.note.transformerlabs.ai/technical-book)

## 📋 10 Deliverables cho Demo Day

| # | Deliverable | File vị trí | Template có sẵn |
|---|-------------|-------------|:---:|
| 1 | Source Code | `src/` | ✅ |
| 2 | README.md | `README_boilerplate.md` → copy thành `README.md` | ✅ |
| 3 | Architecture Diagram | `docs/architecture_diagram.md` | ✅ |
| 4 | AI Logs | LangSmith (3 env vars) + Auto AI Usage Logging | ✅ |
| 5 | Live URL | Deploy lên Render/Vercel | ⚡ CI/CD sẵn |
| 6 | Video Demo | `presentation/` | 📝 |
| 7 | Pitch Deck | `presentation/` | 📝 |
| 8 | Development Journal | `JOURNAL.md` | ✅ |
| 9 | Worklog | `WORKLOG.md` | ✅ |
| 10 | Evaluation Evidence | `eval/` | 📝 |

## 🛠 Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| AI Agent | LangGraph + LangChain | Latest |
| Backend | FastAPI + Uvicorn | 0.100+ |
| LLM | OpenAI GPT-4o-mini | API |
| Frontend | Next.js / Streamlit | 14+ / 1.30+ |
| Database | SQLite (dev) / PostgreSQL (prod) | — |
| DevOps | Docker + GitHub Actions | — |
| Testing | pytest + pytest-asyncio | 8+ |

## 📊 AI Usage Logging

Template đã tích hợp sẵn auto-logging hooks cho 6 AI tools:

| Tool | Cơ chế | Config |
|------|--------|--------|
| Claude Code | `.claude/settings.json` hooks | Tự động |
| Cursor | `.cursor/hooks.json` | Tự động |
| OpenAI Codex CLI | `.codex/hooks.json` | Tự động |
| Gemini CLI | `.gemini/settings.json` | Tự động |
| GitHub Copilot | `.github/hooks/hooks.json` | Tự động |
| Antigravity IDE | Pre-push scan transcript | Tự động trên `git push` |

Tất cả prompts và tool calls được log vào `.ai-log/session.jsonl` và tự động submit lên grading server mỗi khi `git push`.

**ChatGPT / web tools khác** — log thủ công:
```bash
bash scripts/_pyrun.sh scripts/log_manual.py --tool chatgpt --prompt "What you asked"
```

> ⚠️ Chạy `bash scripts/setup_hooks.sh` một lần sau khi clone để cài pre-push hook.

## 📖 Đọc Technical Guidebook

**Online (khuyến nghị):** [phoenix.note.transformerlabs.ai/technical-book](https://phoenix.note.transformerlabs.ai/technical-book)

Đăng nhập bằng GitHub (cùng account đã được BTC mời vào org `AI20K-Build-Cohort-2`)
→ chọn tab **Technical Book** ở sidebar trái → đọc 10 chương + topic sections,
có table of contents bên phải, hỗ trợ light/dark/cyberpunk theme.

**Offline:** mọi chương đều ở thư mục `docs/guide/` trong template này — mở bằng
bất kỳ markdown viewer/editor nào (VS Code, Obsidian, GitHub UI, …).

## 🔗 Liên kết

- 📖 **Technical Guidebook:** [phoenix.note.transformerlabs.ai/technical-book](https://phoenix.note.transformerlabs.ai/technical-book)
- 🏫 **AI20K Program:** VinUni AI20K Build Phase
- 👨‍🏫 **Mentor:** Đặng Hải Lộc

## 📄 License

MIT — Sử dụng tự do cho mục đích giáo dục.
